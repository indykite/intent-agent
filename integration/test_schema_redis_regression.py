"""Regression tests for SchemaClient/_SyncBridge against a *real* Redis server - not in test/,
and not fakeredis, since these bugs only manifest over a genuine socket/event loop. Skips
itself if no real Redis is reachable at REDIS_URL/localhost:6379.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
import redis.exceptions
from redis.asyncio import Redis

from indykite.intent_agent.schema import IkConnection, SchemaClient, app_agent_key_gid
from indykite.redis.storage import RedisStorageBackend

_REAL_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
_SCHEMA = {"graph": {"nodes": {"RegressionCheck": {}}, "edges": []}}
# Bounds the reachability probe below - from_url() defaults both to None (unbounded), so a
# blackholed/unreachable REDIS_URL would hang pytest's own module collection instead of
# resolving to the documented skip.
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
    """A RedisStorageBackend over a genuine redis.asyncio.Redis connection - the whole point of
    this test is to NOT use fakeredis. Deletes its own key afterward so repeated runs don't
    leave test data behind in whatever real Redis is used.

    Cleanup deliberately opens a brand-new client rather than reusing `client`: by teardown
    time, the test body has already run `client` through SchemaClient's own persistent bridge
    loop (see schema.py's _SyncBridge), so its connection is bound to that loop, not whatever
    loop asyncio.run() gives this fixture - the exact class of bug this whole test module
    exists to catch, just one layer up, in the test's own cleanup code.
    """
    app_agent_key = jwt.encode({"sub": "schema-redis-regression-test"}, key="", algorithm="none")
    client = Redis.from_url(_REAL_REDIS_URL)
    backend = RedisStorageBackend(client)
    try:
        yield backend, app_agent_key
    finally:

        async def _cleanup():
            cleanup_client = Redis.from_url(_REAL_REDIS_URL)
            try:
                gid = app_agent_key_gid(app_agent_key)
                await cleanup_client.delete(f"ikg_schema:{gid}")
            finally:
                await cleanup_client.aclose()

        asyncio.run(_cleanup())


def test_get_cached_against_real_redis_does_not_raise_event_loop_closed(mocker, real_redis_backend):
    """get_cached()'s several back-to-back StorageBackend calls (a cache-miss read, a re-check
    read under RefreshGate's lock, then a write of the freshly loaded schema) must all succeed
    against a real Redis connection, reused across every call - this is exactly the sequence
    that used to raise "Event loop is closed".
    """
    backend, app_agent_key = real_redis_backend
    mocker.patch.object(SchemaClient, "load", return_value=_SCHEMA)
    client = SchemaClient(
        IkConnection("https://ikg.example.com", app_agent_key),
        cache=backend,
        refresh_ttl_seconds=300.0,
        executor=ThreadPoolExecutor(max_workers=1),
    )

    result = client.get_cached()

    assert result == _SCHEMA


def test_preload_then_get_cached_against_real_redis_round_trips(mocker, real_redis_backend):
    """The cold-start sequence (preload() at startup, then a later get_cached() on a fresh
    read) must also survive several separate run_sync calls against the same real connection.
    """
    backend, app_agent_key = real_redis_backend
    mocker.patch.object(SchemaClient, "load", return_value=_SCHEMA)
    client = SchemaClient(
        IkConnection("https://ikg.example.com", app_agent_key),
        cache=backend,
        refresh_ttl_seconds=300.0,
        executor=ThreadPoolExecutor(max_workers=1),
    )

    client.preload()
    result = client.get_cached()

    assert result == _SCHEMA


def test_aclose_against_real_redis_does_not_raise_attached_to_a_different_loop(mocker, real_redis_backend):
    """aclose() (called by app.py's lifespan on a different loop than _SyncBridge's own, which
    every prior run_sync-bridged call bound the connection to) must actually close the
    connection, not raise "Task ... got Future ... attached to a different loop" - the same
    class of cross-loop bug this module exists to catch, just in aclose() instead of get_cached().
    """
    backend, app_agent_key = real_redis_backend
    mocker.patch.object(SchemaClient, "load", return_value=_SCHEMA)
    client = SchemaClient(
        IkConnection("https://ikg.example.com", app_agent_key),
        cache=backend,
        refresh_ttl_seconds=300.0,
        executor=ThreadPoolExecutor(max_workers=1),
    )
    client.preload()

    asyncio.run(client.aclose())
