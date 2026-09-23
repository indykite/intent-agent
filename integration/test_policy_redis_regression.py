"""Smoke test for PolicyClient against a *real* Redis server - same reasons (not test/, not
fakeredis) as test_schema_redis_regression.py's module docstring.
"""

from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
import redis

from indykite.intent_agent.policy import PolicyClient, _cache_key
from indykite.intent_agent.schema import CacheConfig, IkConnection
from indykite.redis.storage import RedisStorageBackend
from integration.conftest import REAL_REDIS_URL, real_redis_reachable

_REAL_REDIS_URL = REAL_REDIS_URL
_POLICIES = [{"policy": {"subject": {"type": "Customer"}, "actions": ["RENT"], "resource": {"type": "Car"}}}]
_REFRESH_TTL_SECONDS = 300.0

pytestmark = pytest.mark.skipif(
    not real_redis_reachable(),
    reason=f"no real Redis reachable at {_REAL_REDIS_URL} - see module docstring",
)


@pytest.fixture
def real_redis_backend():
    """A RedisStorageBackend over a genuine redis.Redis connection - see
    test_schema_redis_regression.py's matching fixture.
    """
    app_agent_key = jwt.encode({"sub": "policy-redis-regression-test"}, key="", algorithm="none")
    client = redis.Redis.from_url(_REAL_REDIS_URL)
    backend = RedisStorageBackend(client)
    try:
        yield backend, app_agent_key
    finally:
        cache_key = _cache_key(app_agent_key, "Customer")
        client.delete(cache_key, f"{cache_key}:fresh")
        client.close()


def test_get_cached_against_real_redis_round_trips(mocker, real_redis_backend):
    """get_cached()'s cache-miss read, fetch, and store must all succeed against a real Redis
    connection, and a second call must be a cache hit with no new fetch.
    """
    backend, app_agent_key = real_redis_backend
    load_mock = mocker.patch.object(PolicyClient, "_load", return_value=(_POLICIES, ""))
    client = PolicyClient(
        IkConnection("https://ikg.example.com", app_agent_key),
        cache_config=CacheConfig(
            cache=backend, refresh_ttl_seconds=_REFRESH_TTL_SECONDS, executor=ThreadPoolExecutor(max_workers=1)
        ),
    )

    assert client.get_cached("Customer") == (_POLICIES, False)
    assert client.get_cached("Customer") == (_POLICIES, False)
    assert load_mock.call_count == 1


def test_fresh_marker_gets_a_real_ttl_and_primary_key_does_not(mocker, real_redis_backend):
    """_store_success()'s two-key split must give real EXPIRE/TTL semantics: the primary key has
    no expiry, only the fresh marker does - mirrors SchemaClient's guarantee.
    """
    backend, app_agent_key = real_redis_backend
    mocker.patch.object(PolicyClient, "_load", return_value=(_POLICIES, ""))
    client = PolicyClient(
        IkConnection("https://ikg.example.com", app_agent_key),
        cache_config=CacheConfig(
            cache=backend, refresh_ttl_seconds=_REFRESH_TTL_SECONDS, executor=ThreadPoolExecutor(max_workers=1)
        ),
    )

    client.get_cached("Customer")

    cache_key = _cache_key(app_agent_key, "Customer")
    assert backend._client.ttl(cache_key) == -1  # noqa: SLF001 - no public TTL accessor on StorageBackend
    fresh_ttl = backend._client.ttl(f"{cache_key}:fresh")  # noqa: SLF001 - same
    assert 0 < fresh_ttl <= _REFRESH_TTL_SECONDS
