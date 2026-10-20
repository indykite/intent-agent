"""Starlette application wiring: the IntentAgent, its A2A executor, and the audit sender."""

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import override

from a2a.server.agent_execution import AgentExecutor
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskStore
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

_TASK_ID_NOT_SUPPORTED_MESSAGE = "taskId is not supported"

RefusalHook = Callable[[SendMessageRequest, ServerCallContext, str], Awaitable[None]]


class SendMessageOnlyHandler(DefaultRequestHandler):
    """The SDK's handler, serving SendMessage only, each message as a new task: task reads and
    cancels are unsupported. A message naming a `taskId`, or with invalid subject metadata, is
    refused (and reported to `on_refused` for auditing) before the SDK starts or loads a task -
    so it never reaches the executor, and the SDK doesn't log it as a failed execution.
    """

    def __init__(
        self, *, agent_executor: AgentExecutor, task_store: TaskStore, agent_card: AgentCard, on_refused: RefusalHook
    ) -> None:
        """The SDK's handler over these, reporting refused messages to `on_refused`."""
        super().__init__(agent_executor=agent_executor, task_store=task_store, agent_card=agent_card)
        self._on_refused = on_refused

    @override
    async def on_message_send(self, params: SendMessageRequest, context: ServerCallContext) -> Message | Task:
        try:
            self._validate(params.message)
        except InvalidParamsError as exc:
            await self._on_refused(params, context, exc.message)
            raise
        return await super().on_message_send(params, context)

    @staticmethod
    def _validate(message: Message) -> None:
        """InvalidParamsError for a message naming a `taskId` or with invalid subject metadata."""
        if message.task_id:
            raise InvalidParamsError(message=_TASK_ID_NOT_SUPPORTED_MESSAGE)
        subject_from_message(message)

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

    The audit sender is built from AUDIT_DELIVERY here (or injected, for tests) and closed by
    the app's lifespan on shutdown - which is what drains a buffered (AUDIT_ASYNC) queue
    before the process exits, the same guarantee the IAG's sender gives on stop.

    The lifespan also shuts down this agent's own _fetch_executor (review) - schema/policy
    fetches and their async_creation_runner refreshes all run on it. wait=False,
    cancel_futures=True only drops work still queued and returns immediately rather than
    blocking this coroutine on whatever is already running - it does not, and cannot, stop an
    in-flight synchronous fetch already underway in a worker thread (review: Python cannot
    preempt one, and a genuinely stuck call could still leave that thread running past this
    point, for concurrent.futures' own atexit machinery to join later at interpreter exit - the
    bounded SCHEMA_FETCH_TIMEOUT_SECONDS/POLICY_FETCH_TIMEOUT_SECONDS x retries is what actually
    guarantees such a thread eventually finishes on its own, not this call). What this genuinely
    achieves: the lifespan handler itself never blocks on a slow/stuck fetch, and, since each
    IntentAgent owns its own executor rather than sharing one process-wide singleton, this only
    ever touches the instance this specific app is backed by - a second app (or any other
    embedded caller) built on its own IntentAgent is unaffected.
    """
    agent = IntentAgent(settings, llm=llm, schema_client=schema_client, policy_client=policy_client)
    agent.preload_schema()
    sender = audit_sender or build_sender(settings)

    agent_card = agent.build_agent_card()
    auth.add_bearer_security(agent_card, settings.expected_token_audience)
    executor = IntentExecutor(agent, sender)
    handler = SendMessageOnlyHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
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
