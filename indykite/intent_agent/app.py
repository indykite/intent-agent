"""Starlette application wiring: the IntentAgent, its A2A executor, and the audit sender."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import override

from a2a.server.agent_execution import AgentExecutor
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    GetTaskRequest,
    ListTasksRequest,
    ListTasksResponse,
    Message,
    SendMessageRequest,
    Task,
)
from a2a.utils.constants import DEFAULT_RPC_URL
from a2a.utils.errors import InvalidParamsError, UnsupportedOperationError
from langchain_core.language_models import BaseChatModel
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route

from indykite.intent_agent import auth
from indykite.intent_agent.agent import IntentAgent
from indykite.intent_agent.audit.sender import Sender, build_sender
from indykite.intent_agent.config import Settings
from indykite.intent_agent.executor import IntentExecutor, subject_from_message
from indykite.intent_agent.policy import PolicyClient
from indykite.intent_agent.schema import SchemaClient

logger = logging.getLogger(__name__)

_TASK_ID_NOT_SUPPORTED_MESSAGE = "taskId is not supported"
_RETURN_IMMEDIATELY_NOT_SUPPORTED_MESSAGE = "returnImmediately is not supported"
# ServerCallContext.state key: the ids of the tasks a request saved.
_SAVED_TASK_IDS = "intent_agent.saved_task_ids"

RefusalHook = Callable[[SendMessageRequest, ServerCallContext, str], Awaitable[None]]


class RequestScopedTaskStore(InMemoryTaskStore):
    """The SDK's in-memory task store, emptied of a request's tasks once it is answered: tasks
    can't be read or continued, so keeping them would only grow memory with every request.
    """

    @override
    async def save(self, task: Task, context: ServerCallContext) -> None:
        context.state.setdefault(_SAVED_TASK_IDS, set()).add(task.id)
        await super().save(task, context)

    async def forget(self, context: ServerCallContext) -> None:
        """Delete every task saved under this request's context; never raises, so a failed delete
        can't change the response.
        """
        for task_id in context.state.pop(_SAVED_TASK_IDS, set()):
            try:
                await self.delete(task_id, context)
            except Exception:
                logger.exception("Failed to delete task %s", task_id)


class SendMessageOnlyHandler(DefaultRequestHandler):
    """The SDK's handler, serving SendMessage only, each message as a new task.

    A message naming a `taskId`, asking for `returnImmediately`, or with invalid subject metadata,
    is refused before any task starts, so it never reaches the executor, and is reported to
    `on_refused` for auditing. A task is deleted once its work has finished, even when the request
    itself was cancelled first.
    """

    def __init__(
        self,
        *,
        agent_executor: AgentExecutor,
        task_store: RequestScopedTaskStore,
        agent_card: AgentCard,
        on_refused: RefusalHook,
    ) -> None:
        """The SDK's handler over these, reporting refused messages to `on_refused`."""
        super().__init__(agent_executor=agent_executor, task_store=task_store, agent_card=agent_card)
        self._tasks = task_store
        self._on_refused = on_refused
        self._pending_cleanups: set[asyncio.Task[None]] = set()

    @override
    async def on_message_send(self, params: SendMessageRequest, context: ServerCallContext) -> Message | Task:
        try:
            self._validate(params)
        except InvalidParamsError as exc:
            await self._on_refused(params, context, exc.message)
            raise
        # Shielded, so a cancelled request doesn't stop the wait for the work it started: the
        # agent keeps running and saving the task, and the cleanup has to come after that.
        work = asyncio.ensure_future(super().on_message_send(params, context))
        try:
            return await asyncio.shield(work)
        finally:
            if work.done():
                await self._tasks.forget(context)
            else:
                work.add_done_callback(lambda _: self._forget_later(context))

    def _forget_later(self, context: ServerCallContext) -> None:
        """Delete a cancelled request's tasks once its work has finished."""
        cleanup = asyncio.create_task(self._tasks.forget(context))
        self._pending_cleanups.add(cleanup)
        cleanup.add_done_callback(self._pending_cleanups.discard)

    @staticmethod
    def _validate(params: SendMessageRequest) -> None:
        """InvalidParamsError for a message naming a `taskId`, asking for `returnImmediately`, or
        with invalid subject metadata.
        """
        if params.message.task_id:
            raise InvalidParamsError(message=_TASK_ID_NOT_SUPPORTED_MESSAGE)
        if params.configuration.return_immediately:
            raise InvalidParamsError(message=_RETURN_IMMEDIATELY_NOT_SUPPORTED_MESSAGE)
        subject_from_message(params.message)

    @override
    async def on_get_task(self, params: GetTaskRequest, context: ServerCallContext) -> Task | None:
        raise UnsupportedOperationError(message="GetTask is not supported.")

    @override
    async def on_list_tasks(self, params: ListTasksRequest, context: ServerCallContext) -> ListTasksResponse:
        raise UnsupportedOperationError(message="ListTasks is not supported.")

    @override
    async def on_cancel_task(self, params: CancelTaskRequest, context: ServerCallContext) -> Task | None:
        raise UnsupportedOperationError(message="CancelTask is not supported.")


def build_app(
    settings: Settings,
    *,
    llm: BaseChatModel | None = None,
    schema_client: SchemaClient | None = None,
    policy_client: PolicyClient | None = None,
    audit_sender: Sender | None = None,
) -> tuple[Starlette, IntentAgent]:
    """Construct the Starlette app and its backing IntentAgent for these settings.

    On shutdown the lifespan closes the audit sender (draining an async queue) and cancels
    queued work on this agent's fetch executor without waiting for in-flight fetches.
    """
    agent = IntentAgent(settings, llm=llm, schema_client=schema_client, policy_client=policy_client)
    agent.preload_schema()
    sender = audit_sender or build_sender(settings)

    agent_card = agent.build_agent_card()
    auth.add_bearer_security(agent_card, settings.expected_token_audience)
    executor = IntentExecutor(agent, sender)
    handler = SendMessageOnlyHandler(
        agent_executor=executor,
        task_store=RequestScopedTaskStore(),
        agent_card=agent_card,
        on_refused=executor.audit_refused_message,
    )
    # Only the JSON-RPC route: the agent card stays public.
    authenticated = Middleware(
        auth.BearerAuthMiddleware,
        config=auth.IntrospectionConfig(
            introspect_url=str(settings.introspect_url),
            client_id=settings.introspect_client_id,
            client_secret=settings.introspect_client_secret.get_secret_value(),
            expected_audience=settings.expected_token_audience,
            timeout_seconds=settings.introspect_timeout_seconds,
        ),
        on_rejected=executor.audit_rejected_request,
    )
    rpc_routes = [
        Route(route.path, route.endpoint, methods=route.methods, name=route.name, middleware=[authenticated])
        for route in create_jsonrpc_routes(
            request_handler=handler, rpc_url=DEFAULT_RPC_URL, context_builder=auth.CallContextBuilder()
        )
    ]

    @contextlib.asynccontextmanager
    async def _lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            agent._fetch_executor.shutdown(wait=False, cancel_futures=True)  # noqa: SLF001 - this app's own agent
            await sender.close()

    app = Starlette(
        routes=[
            *create_agent_card_routes(agent_card=agent_card),
            *rpc_routes,
        ],
        lifespan=_lifespan,
    )
    return app, agent
