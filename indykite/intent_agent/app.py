"""Starlette application wiring: the IntentAgent, its A2A executor, and the audit sender."""

import contextlib
from collections.abc import AsyncIterator

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskStore
from a2a.utils.constants import DEFAULT_RPC_URL
from langchain_core.language_models import BaseChatModel
from starlette.applications import Starlette

from indykite.a2a.redis.queue_manager import RedisQueueManager
from indykite.a2a.redis.task_store import RedisTaskStore
from indykite.intent_agent.agent import IntentAgent
from indykite.intent_agent.audit.sender import Sender, build_sender
from indykite.intent_agent.config import Settings, StorageBackendKind
from indykite.intent_agent.executor import IntentExecutor
from indykite.intent_agent.policy import PolicyClient
from indykite.intent_agent.schema import SchemaClient, app_agent_key_gid
from indykite.redis.storage import RedisStorageBackend, build_redis_client

_SECONDS_PER_MINUTE = 60
_MILLISECONDS_PER_SECOND = 1000


def _build_task_store_and_queue_manager(settings: Settings) -> tuple[TaskStore, RedisQueueManager | None]:
    """A2A_TASK_STORE_BACKEND picks memory (InMemoryTaskStore, no queue manager) or the
    Redis-backed pair below, sharing one client. Never imports redis.asyncio directly - see
    schema.build_schema_backend's matching docstring.
    """
    if settings.a2a_task_store_backend != StorageBackendKind.REDIS:
        return InMemoryTaskStore(), None
    task_store_client = build_redis_client(
        str(settings.a2a_task_store_url), socket_timeout_seconds=settings.redis_socket_timeout_seconds
    )
    env_id = app_agent_key_gid(settings.ik_app_agent_key.get_secret_value()) or "unconfigured"
    task_store = RedisTaskStore(
        RedisStorageBackend(task_store_client),
        env_id=env_id,
        terminal_ttl_seconds=settings.a2a_task_store_ttl_minutes * _SECONDS_PER_MINUTE,
    )
    # queue_manager wiring is currently inert - see queue_manager.py's module docstring.
    return task_store, RedisQueueManager(
        task_store_client,
        env_id=env_id,
        dequeue_block_milliseconds=int(settings.redis_queue_dequeue_block_seconds * _MILLISECONDS_PER_SECOND),
        reclaim_min_idle_milliseconds=int(settings.redis_queue_reclaim_min_idle_seconds * _MILLISECONDS_PER_SECOND),
    )


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
    task_store, queue_manager = _build_task_store_and_queue_manager(settings)
    handler = DefaultRequestHandler(
        agent_executor=IntentExecutor(agent, sender),
        task_store=task_store,
        queue_manager=queue_manager,
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
            # InMemoryTaskStore has nothing to close; RedisTaskStore/RedisQueueManager share
            # one client, so closing the store's closes both.
            if isinstance(task_store, RedisTaskStore):
                await task_store.aclose()

    app = Starlette(
        routes=[
            *create_agent_card_routes(agent_card=agent_card),
            *create_jsonrpc_routes(request_handler=handler, rpc_url=DEFAULT_RPC_URL),
        ],
        lifespan=_lifespan,
    )
    return app, agent
