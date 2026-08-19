"""Tests for SchemaClient: fetching, caching, and TTL-driven on-demand refresh."""

import httpx2
import pytest
from dogpile.cache.region import make_region

from indykite.intent_agent.schema import SchemaClient

_SCHEMA = {"graph": {"nodes": {}, "edges": []}}
_EXPECTED_FETCH_COUNT_AFTER_REFRESH = 2


def _memory_cache():
    """Build a fresh in-process Memory-backed CacheRegion for a test's SchemaClient."""
    return make_region().configure("dogpile.cache.memory")


def _mock_sync_client(mocker, *, response=None, side_effect=None):
    """Patch httpx2.Client so `with httpx2.Client(...) as client: client.get(...)` yields/raises as given."""
    client = mocker.MagicMock()
    client.__enter__ = mocker.Mock(return_value=client)
    client.__exit__ = mocker.Mock(return_value=False)
    client.get = mocker.Mock(return_value=response, side_effect=side_effect)
    mocker.patch("indykite.intent_agent.schema.httpx2.Client", return_value=client)
    return client


def _mock_response(mocker):
    """Build a fake httpx2 response returning _SCHEMA from .json(), never raising on .raise_for_status()."""
    response = mocker.Mock(json=mocker.Mock(return_value=_SCHEMA))
    response.raise_for_status = mocker.Mock()
    return response


def _expire(cache):
    """Force every entry in this Memory-backed CacheRegion to read as expired, deterministically -
    backdates the stored CachedValue's creation time instead of sleeping past expiration_time,
    which would make the test slow and racy against real wall-clock timing.
    """
    for key, cached in cache.backend._cache.items():
        cache.backend._cache[key] = cached._replace(metadata={**cached.metadata, "ct": 0})


def _mock_redis_client(mocker):
    """Patch dogpile's Redis backend to use an in-memory dict instead of a real connection -
    exercises the real CacheRegion -> RedisBackend -> serialize/deserialize path, only the
    network client itself is faked, no real Redis server needed.
    """
    store: dict[bytes, bytes] = {}
    client = mocker.Mock(
        get=mocker.Mock(side_effect=store.get),
        set=mocker.Mock(side_effect=store.__setitem__),
    )
    mocker.patch("redis.StrictRedis", return_value=client)
    return client


@pytest.mark.parametrize(
    ("base_url", "app_agent_key"),
    [
        pytest.param("", "app-key", id="blank_base_url"),
        pytest.param("https://ikg.example.com", "", id="blank_app_agent_key"),
    ],
)
def test_load_returns_none_when_config_incomplete(base_url, app_agent_key):
    """Missing base_url or app_agent_key means schema unavailable, no request attempted."""
    client = SchemaClient(base_url, app_agent_key, cache=_memory_cache())
    assert client.load() is None


def test_load_returns_none_on_http_error(mocker):
    """An HTTP error from the platform must return None, not raise."""
    _mock_sync_client(mocker, side_effect=httpx2.HTTPError("boom"))
    client = SchemaClient("https://ikg.example.com", "app-key", cache=_memory_cache())
    assert client.load() is None


def test_load_returns_schema_on_success(mocker):
    """A successful fetch returns the parsed JSON schema."""
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", "app-key", cache=_memory_cache())
    assert client.load() == _SCHEMA


@pytest.mark.parametrize("backend", ["memory", "redis"])
def test_preload_and_get_cached_round_trip(mocker, backend):
    """preload() then get_cached() must return the same schema, on both the default in-process
    Memory cache and a Redis-backed CacheRegion (the real dogpile Redis backend - only its
    network client is faked, so this exercises the actual serialize/deserialize path).
    """
    if backend == "redis":
        _mock_redis_client(mocker)
        cache = make_region().configure("dogpile.cache.redis", arguments={"host": "localhost", "port": 6379})
    else:
        cache = _memory_cache()
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", "app-key", cache=cache)

    client.preload()

    assert client.get_cached() == _SCHEMA


def test_get_cached_fetches_on_demand_when_not_already_cached(mocker):
    """get_cached() must fetch-and-cache on its own when nothing is cached yet: either because
    preload() was never called (fetch succeeds) or because there's no config to fetch with at
    all (fetch fails, returns None instead of raising).
    """
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", "app-key", cache=_memory_cache())
    assert client.get_cached() == _SCHEMA

    unconfigured_client = SchemaClient("", "", cache=_memory_cache())
    assert unconfigured_client.get_cached() is None


def test_get_cached_refreshes_on_demand_when_entry_expired(mocker):
    """An expired cache entry must trigger a fresh load() and re-cache, not just sit stale."""
    cache = make_region().configure("dogpile.cache.memory", expiration_time=100)
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", "app-key", cache=cache)
    client.preload()
    assert client_mock.get.call_count == 1

    _expire(cache)

    assert client.get_cached() == _SCHEMA
    assert client_mock.get.call_count == _EXPECTED_FETCH_COUNT_AFTER_REFRESH


def test_get_cached_falls_back_to_stale_schema_when_refresh_fails(mocker, caplog):
    """Per TDD 4.4: if the on-demand refresh fails, the expired-but-last-known-good schema
    must still be served (not None) and schema_refresh_failed logged - a transient platform
    outage must not take down an already-running agent.
    """
    caplog.set_level("ERROR")
    cache = make_region().configure("dogpile.cache.memory", expiration_time=100)
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", "app-key", cache=cache)
    client.preload()
    _expire(cache)

    client_mock.get.side_effect = httpx2.HTTPError("platform unreachable")

    assert client.get_cached() == _SCHEMA
    assert any("schema_refresh_failed" in record.getMessage() for record in caplog.records)
