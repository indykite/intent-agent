"""Smoke tests for SchemaClient against a *real* Redis server - not fakeredis, since real
EXPIRE/TTL wire semantics are worth confirming beyond what it emulates. Skips if unreachable.
"""

from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
import redis

from indykite.intent_agent.schema import CacheConfig, IkConnection, SchemaClient, env_scope
from indykite.redis.storage import RedisStorageBackend
from integration.conftest import REAL_REDIS_URL, real_redis_reachable

_REAL_REDIS_URL = REAL_REDIS_URL
_SCHEMA = {"graph": {"nodes": {"RegressionCheck": {}}, "edges": []}}
_REFRESH_TTL_SECONDS = 300.0

pytestmark = pytest.mark.skipif(
    not real_redis_reachable(),
    reason=f"no real Redis reachable at {_REAL_REDIS_URL} - see module docstring",
)


@pytest.fixture
def real_redis_backend():
    """A RedisStorageBackend over a genuine redis.Redis connection, not fakeredis. Deletes its
    own keys afterward so repeated runs don't leave test data behind.
    """
    app_agent_key = jwt.encode({"sub": "schema-redis-regression-test"}, key="", algorithm="none")
    client = redis.Redis.from_url(_REAL_REDIS_URL)
    backend = RedisStorageBackend(client)
    try:
        yield backend, app_agent_key
    finally:
        scope = env_scope(app_agent_key)
        client.delete(f"ikg:{scope}:schema", f"ikg:{scope}:schema:fresh")
        client.close()


def test_get_cached_against_real_redis_round_trips(mocker, real_redis_backend):
    """get_cached()'s back-to-back StorageBackend calls (cache-miss read, re-check under
    RefreshGate's lock, then write) must all succeed against a real Redis connection.
    """
    backend, app_agent_key = real_redis_backend
    mocker.patch.object(SchemaClient, "load", return_value=_SCHEMA)
    client = SchemaClient(
        IkConnection("https://ikg.example.com", app_agent_key),
        cache_config=CacheConfig(
            cache=backend, refresh_ttl_seconds=_REFRESH_TTL_SECONDS, executor=ThreadPoolExecutor(max_workers=1)
        ),
    )

    result = client.get_cached()

    assert result == _SCHEMA


def test_preload_then_get_cached_against_real_redis_round_trips(mocker, real_redis_backend):
    """The cold-start sequence (preload() at startup, then a later get_cached() on a fresh
    read) must also survive several separate calls against the same real connection.
    """
    backend, app_agent_key = real_redis_backend
    mocker.patch.object(SchemaClient, "load", return_value=_SCHEMA)
    client = SchemaClient(
        IkConnection("https://ikg.example.com", app_agent_key),
        cache_config=CacheConfig(
            cache=backend, refresh_ttl_seconds=_REFRESH_TTL_SECONDS, executor=ThreadPoolExecutor(max_workers=1)
        ),
    )

    client.preload()
    result = client.get_cached()

    assert result == _SCHEMA


def test_fresh_marker_gets_a_real_ttl_and_primary_key_does_not(mocker, real_redis_backend):
    """_store()'s two-key split must hold under real EXPIRE/TTL semantics, not just fakeredis's
    emulation: the primary key survives with no expiry, only the fresh marker carries one.
    """
    backend, app_agent_key = real_redis_backend
    mocker.patch.object(SchemaClient, "load", return_value=_SCHEMA)
    client = SchemaClient(
        IkConnection("https://ikg.example.com", app_agent_key),
        cache_config=CacheConfig(
            cache=backend, refresh_ttl_seconds=_REFRESH_TTL_SECONDS, executor=ThreadPoolExecutor(max_workers=1)
        ),
    )

    client.preload()

    scope = env_scope(app_agent_key)
    assert backend._client.ttl(f"ikg:{scope}:schema") == -1  # noqa: SLF001 - no public TTL accessor on StorageBackend
    fresh_ttl = backend._client.ttl(f"ikg:{scope}:schema:fresh")  # noqa: SLF001 - same
    assert 0 < fresh_ttl <= _REFRESH_TTL_SECONDS
