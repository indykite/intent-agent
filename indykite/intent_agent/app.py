"""Starlette application wiring: the IntentAgent, its A2A executor, and the audit sender."""

import contextlib
from collections.abc import AsyncIterator

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.utils.constants import DEFAULT_RPC_URL
from langchain_core.language_models import BaseChatModel
from starlette.applications import Starlette

from indykite.intent_agent.agent import IntentAgent
from indykite.intent_agent.audit.sender import Sender, build_sender
from indykite.intent_agent.config import Settings
from indykite.intent_agent.executor import IntentExecutor
from indykite.intent_agent.policy import PolicyClient
from indykite.intent_agent.schema import SchemaClient


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
    fetches and their async_creation_runner refreshes all run on it, and Python cannot preempt
    an already-running synchronous call, so a stuck one could otherwise hang process shutdown
    waiting for it via the executor's own atexit handler. wait=False, cancel_futures=True drops
    anything still queued and does not block on whatever is already running - acceptable here
    since the whole process is exiting anyway, not something a live request would ever see.
    Each IntentAgent owns its own executor rather than sharing one process-wide singleton, so
    this only ever tears down the instance this specific app is backed by - a second app (or
    any other embedded caller) built on its own IntentAgent is unaffected.
    """
    agent = IntentAgent(settings, llm=llm, schema_client=schema_client, policy_client=policy_client)
    agent.preload_schema()
    sender = audit_sender or build_sender(settings)

    agent_card = agent.build_agent_card()
    handler = DefaultRequestHandler(
        agent_executor=IntentExecutor(agent, sender),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

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
            *create_jsonrpc_routes(request_handler=handler, rpc_url=DEFAULT_RPC_URL),
        ],
        lifespan=_lifespan,
    )
    return app, agent
