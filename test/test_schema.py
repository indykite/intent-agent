"""Tests for SchemaClient: fetching, caching, and TTL-based background refresh."""

import httpx2
import pytest
from dogpile.cache.region import make_region

from indykite.intent_agent.schema import SchemaClient

_SCHEMA = {"graph": {"nodes": {}, "edges": []}}


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
    client = SchemaClient(base_url, app_agent_key)
    assert client.load() is None


def test_load_returns_none_on_http_error(mocker):
    """An HTTP error from the platform must return None, not raise."""
    _mock_sync_client(mocker, side_effect=httpx2.HTTPError("boom"))
    client = SchemaClient("https://ikg.example.com", "app-key")
    assert client.load() is None


def test_load_returns_schema_on_success(mocker):
    """A successful fetch returns the parsed JSON schema."""
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", "app-key")
    assert client.load() == _SCHEMA


@pytest.mark.parametrize("backend", ["memory", "redis"])
def test_preload_and_get_cached_round_trip(mocker, backend):
    """preload() then get_cached() must return the same schema, on both the default in-process
    Memory cache and a Redis-backed CacheRegion (the real dogpile Redis backend - only its
    network client is faked, so this exercises the actual serialize/deserialize path).
    """
    cache = None
    if backend == "redis":
        _mock_redis_client(mocker)
        cache = make_region().configure("dogpile.cache.redis", arguments={"host": "localhost", "port": 6379})
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
    client = SchemaClient("https://ikg.example.com", "app-key")
    assert client.get_cached() == _SCHEMA

    unconfigured_client = SchemaClient("", "")
    assert unconfigured_client.get_cached() is None


def test_refresh_updates_cache_on_success(mocker):
    """A successful refresh must update the cache."""
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", "app-key")

    client.refresh()

    assert client.get_cached() == _SCHEMA


def test_refresh_keeps_last_good_schema_on_failure(mocker, caplog):
    """Per TDD 4.4: a failed refresh must not clear the cache, and must log schema_refresh_failed."""
    caplog.set_level("ERROR")
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", "app-key")
    client.preload()

    client_mock.get.side_effect = httpx2.HTTPError("platform unreachable")
    client.refresh()

    assert client.get_cached() == _SCHEMA
    assert any("schema_refresh_failed" in record.getMessage() for record in caplog.records)


def test_start_background_refresh_runs_on_daemon_thread(mocker):
    """The refresh loop must run on a background daemon thread, not block the caller."""
    thread_cls = mocker.patch("indykite.intent_agent.schema.threading.Thread")
    client = SchemaClient("https://ikg.example.com", "app-key")

    client.start_background_refresh(60.0)

    assert thread_cls.call_args.kwargs["daemon"] is True
    thread_cls.return_value.start.assert_called_once()
