"""Regression test for PolicyClient.aclose() against a *real* Redis server - same bug class and
same reasons (not test/, not fakeredis) as test_schema_redis_regression.py's module docstring.
"""

import asyncio
import os

import jwt
import pytest
import redis.exceptions
from redis.asyncio import Redis

from indykite.intent_agent.policy import PolicyClient, _cache_key
from indykite.intent_agent.schema import IkConnection
from indykite.redis.storage import RedisStorageBackend

_REAL_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
_POLICIES = [{"policy": {"subject": {"type": "Customer"}, "actions": ["RENT"], "resource": {"type": "Car"}}}]
# Bounds the reachability probe below - see test_schema_redis_regression.py's matching constant.
_PROBE_TIMEOUT_SECONDS = 2.0


def _real_redis_reachable() -> bool:
    """True if a real Redis actually answers PING at _REAL_REDIS_URL."""
    try:
        client = redis.Redis.from_url(
            _REAL_REDIS_URL, socket_timeout=_PROBE_TIMEOUT_SECONDS, socket_connect_timeout=_PROBE_TIMEOUT_SECONDS
        )
        try:
            return client.ping()
        finally:
            client.close()
    except redis.exceptions.RedisError:
        return False


pytestmark = pytest.mark.skipif(
    not _real_redis_reachable(),
    reason=f"no real Redis reachable at {_REAL_REDIS_URL} - see module docstring",
)


@pytest.fixture
def real_redis_backend():
    """A RedisStorageBackend over a genuine redis.asyncio.Redis connection - see
    test_schema_redis_regression.py's matching fixture for why cleanup opens a brand-new client.
    """
    app_agent_key = jwt.encode({"sub": "policy-redis-regression-test"}, key="", algorithm="none")
    client = Redis.from_url(_REAL_REDIS_URL)
    backend = RedisStorageBackend(client)
    try:
        yield backend, app_agent_key
    finally:

        async def _cleanup():
            cleanup_client = Redis.from_url(_REAL_REDIS_URL)
            try:
                await cleanup_client.delete(_cache_key(app_agent_key, "Customer"))
            finally:
                await cleanup_client.aclose()

        asyncio.run(_cleanup())


def test_aclose_against_real_redis_does_not_raise_attached_to_a_different_loop(mocker, real_redis_backend):
    """aclose() must actually close the connection, not raise "Task ... got Future ... attached
    to a different loop" - see module docstring.
    """
    backend, app_agent_key = real_redis_backend
    mocker.patch.object(PolicyClient, "_load", return_value=(_POLICIES, None))
    client = PolicyClient(
        IkConnection("https://ikg.example.com", app_agent_key), cache=backend, refresh_ttl_seconds=300.0
    )
    client.get_cached("Customer")

    asyncio.run(client.aclose())
