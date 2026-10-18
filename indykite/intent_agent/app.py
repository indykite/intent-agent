"""Starlette application wiring: the IntentAgent, its A2A executor, and the audit sender."""

import contextlib
from collections.abc import AsyncIterator

from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.types.a2a_pb2 import CancelTaskRequest, GetTaskRequest, ListTasksRequest, ListTasksResponse, Task
from a2a.utils.constants import DEFAULT_RPC_URL
from a2a.utils.errors import UnsupportedOperationError
from langchain_core.language_models import BaseChatModel
from starlette.applications import Starlette

from indykite.a2a.task_store import EvictingTaskStore
from indykite.intent_agent.agent import IntentAgent
from indykite.intent_agent.audit.sender import Sender, build_sender
from indykite.intent_agent.config import Settings
from indykite.intent_agent.executor import IntentExecutor
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
    task_store = EvictingTaskStore(terminal_ttl_seconds=settings.task_store_terminal_ttl_minutes * _SECONDS_PER_MINUTE)
    handler = _SendMessageOnlyRequestHandler(
        agent_executor=IntentExecutor(agent, sender),
        task_store=task_store,
        agent_card=agent_card,
    )

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
            *create_jsonrpc_routes(request_handler=handler, rpc_url=DEFAULT_RPC_URL),
        ],
        lifespan=_lifespan,
    )
    return app, agent
