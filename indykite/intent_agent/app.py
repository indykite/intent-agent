"""Starlette application wiring: the IntentAgent, its A2A executor, and the audit sender."""

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import override

from a2a.server.agent_execution import AgentExecutor
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import TaskStore
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

from indykite.a2a.task_store import EvictingTaskStore
from indykite.intent_agent import auth
from indykite.intent_agent.agent import IntentAgent
from indykite.intent_agent.audit.sender import Sender, build_sender
from indykite.intent_agent.config import Settings
from indykite.intent_agent.executor import IntentExecutor, subject_from_message
from indykite.intent_agent.policy import PolicyClient
from indykite.intent_agent.schema import SchemaClient

_SECONDS_PER_MINUTE = 60


class _SendMessageOnlyRequestHandler(DefaultRequestHandler):
    """IAG's only contract is SendMessage (result/error embedded in that same response, a task
    never looked up or canceled later) - GetTask/ListTasks/CancelTask are still A2A-exposed, so
    reject them explicitly instead of leaving the task store's internal get()/list() to answer.
    """

    async def on_get_task(self, _params: GetTaskRequest, _context: ServerCallContext) -> Task | None:
        raise UnsupportedOperationError(message="GetTask is not supported.")

    async def on_list_tasks(self, _params: ListTasksRequest, _context: ServerCallContext) -> ListTasksResponse:
        raise UnsupportedOperationError(message="ListTasks is not supported.")

    async def on_cancel_task(self, _params: CancelTaskRequest, _context: ServerCallContext) -> Task | None:
        raise UnsupportedOperationError(message="CancelTask is not supported.")

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
    the app's lifespan on shutdown, draining any buffered queue before the process exits.

    The lifespan also shuts down this agent's own _fetch_executor. wait=False,
    cancel_futures=True drops queued work and returns immediately rather than blocking on
    what's already running - it cannot stop an in-flight fetch already underway in a worker
    thread (Python can't preempt one), but the bounded fetch timeouts x retries guarantee such
    a thread finishes on its own eventually.
    """
    agent = IntentAgent(settings, llm=llm, schema_client=schema_client, policy_client=policy_client)
    agent.preload_schema()
    sender = audit_sender or build_sender(settings)

    agent_card = agent.build_agent_card()
    auth.add_bearer_security(agent_card, settings.expected_token_audience)
    executor = IntentExecutor(agent, sender)
    task_store = EvictingTaskStore(terminal_ttl_seconds=settings.task_store_terminal_ttl_minutes * _SECONDS_PER_MINUTE)
    handler = SendMessageOnlyHandler(
        agent_executor=executor,
        task_store=task_store,
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
            await agent.aclose()

    app = Starlette(
        routes=[
            *create_agent_card_routes(agent_card=agent_card),
            *rpc_routes,
        ],
        lifespan=_lifespan,
    )
    return app, agent
