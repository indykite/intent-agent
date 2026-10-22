"""Tests for SchemaClient: fetching, caching, and TTL-driven on-demand refresh."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx2
import jwt
import pytest
from dogpile.cache.region import make_region

from indykite.intent_agent.errors import InvalidAppAgentKeyError
from indykite.intent_agent.schema import SchemaClient, _cache_key, _make_async_creation_runner

_SCHEMA = {"graph": {"nodes": {}, "edges": []}}
_EXPECTED_FETCH_COUNT_AFTER_REFRESH = 2


def _fake_jwt(sub):
    """An unsigned JWT carrying `sub`, encoded with the same library that decodes it."""
    return jwt.encode({"sub": sub}, key="", algorithm="none")


# SchemaClient requires a well-formed JWT for any non-blank app_agent_key.
_APP_AGENT_KEY = _fake_jwt("test-app-agent")


def _memory_cache():
    """A fresh in-memory CacheRegion."""
    return make_region().configure("dogpile.cache.memory")


def _mock_sync_client(mocker, *, response=None, side_effect=None):
    """Patch httpx2.Client so `with httpx2.Client(...) as client: client.get(...)` yields/raises as given."""
    client = mocker.MagicMock()
    client.__enter__ = mocker.Mock(return_value=client)
    client.__exit__ = mocker.Mock(return_value=False)
    client.get = mocker.Mock(return_value=response, side_effect=side_effect)
    mocker.patch("indykite.intent_agent.schema.httpx2.Client", return_value=client)
    return client


def _mock_response(mocker, schema=None):
    """A fake 200 response whose .json() is `schema`, defaulting to _SCHEMA."""
    if schema is None:
        schema = _SCHEMA
    response = mocker.Mock(json=mocker.Mock(return_value=schema))
    response.status_code = 200
    response.text = json.dumps(schema)
    response.raise_for_status = mocker.Mock()
    return response


def _expire(cache):
    """Backdate every cached entry's creation time so dogpile treats it as expired."""
    # Not zero: dogpile reads a creation time of 0 as "never created", not "stale".
    for key, cached in cache.backend._cache.items():
        expired_ct = time.time() - 1_000_000
        cache.backend._cache[key] = cached._replace(metadata={**cached.metadata, "ct": expired_ct})


def _memory_cache_with_async_refresh():
    """An in-memory CacheRegion with the real async refresh runner, plus its executor to wait on."""
    executor = ThreadPoolExecutor(max_workers=1)
    cache = make_region(async_creation_runner=_make_async_creation_runner(executor)).configure(
        "dogpile.cache.memory", expiration_time=100
    )
    return cache, executor


def _wait_for_async_refresh(executor):
    """Block until every refresh already submitted to `executor` has finished."""
    executor.shutdown(wait=True)


def _mock_redis_client(mocker):
    """Replace dogpile's Redis network client with an in-memory dict."""
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
        pytest.param("", _APP_AGENT_KEY, id="blank_base_url"),
        pytest.param("https://ikg.example.com", "", id="blank_app_agent_key"),
    ],
)
def test_load_returns_none_when_config_incomplete(base_url, app_agent_key):
    """Missing base_url or app_agent_key means schema unavailable, no request attempted."""
    client = SchemaClient(base_url, app_agent_key, cache=_memory_cache())
    assert client.load() is None


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(httpx2.ConnectTimeout("connect timed out"), id="connection_failure"),
        pytest.param(httpx2.HTTPStatusError("401 Unauthorized", request=None, response=None), id="bad_credential"),
    ],
)
def test_load_returns_none_on_http_error(mocker, caplog, error):
    """An HTTP error returns None and its specific cause is logged."""
    caplog.set_level("ERROR")
    _mock_sync_client(mocker, side_effect=error)
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())
    assert client.load() is None
    assert str(error) in caplog.text


def _mock_malformed_json_response(mocker):
    """A 2xx response whose .json() raises json.JSONDecodeError."""
    response = mocker.Mock()
    response.status_code = 200
    response.text = "not json"
    response.raise_for_status = mocker.Mock()
    response.json = mocker.Mock(side_effect=json.JSONDecodeError("bad json", "not json", 0))
    return response


def test_load_returns_none_on_malformed_json_response(mocker, caplog):
    """A 2xx response with a non-JSON body returns None like an HTTP error."""
    caplog.set_level("ERROR")
    _mock_sync_client(mocker, response=_mock_malformed_json_response(mocker))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())
    assert client.load() is None
    assert "Failed to load data schema" in caplog.text


def test_load_returns_schema_on_success(mocker):
    """A successful fetch returns the parsed JSON schema."""
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())
    assert client.load() == _SCHEMA


def test_load_logs_the_request_url_and_a_truncated_response_preview(mocker, caplog):
    """The request URL, response status and a body[:500] preview are logged at DEBUG."""
    caplog.set_level("DEBUG")
    large_schema = {"graph": {"nodes": {"Padding": {"description": "x" * 600}}, "edges": []}}
    _mock_sync_client(mocker, response=_mock_response(mocker, schema=large_schema))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())

    assert client.load() == large_schema

    assert "Requesting schema from https://ikg.example.com/data-schema/v1" in caplog.text
    full_body = json.dumps(large_schema)
    assert len(full_body) > len(full_body[:500])
    assert f"Schema response: status=200 body[:500]={full_body[:500]!r}" in caplog.text
    assert full_body not in caplog.text


@pytest.mark.parametrize("backend", ["memory", "redis"])
def test_preload_and_get_cached_round_trip(mocker, caplog, backend):
    """preload() then get_cached() round-trips the schema on both backends and logs a cache hit."""
    caplog.set_level("DEBUG")
    if backend == "redis":
        _mock_redis_client(mocker)
        cache = make_region().configure("dogpile.cache.redis", arguments={"host": "localhost", "port": 6379})
    else:
        cache = _memory_cache()
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    client.preload()

    assert client.get_cached() == _SCHEMA
    assert f"cache_key={client._cache_key} hit=True" in caplog.text


def test_cache_key_uses_jwt_sub_claim_blank_placeholder_or_raises(caplog):
    """The key uses the JWT's sub, a placeholder when blank, and raises (logging why) otherwise."""
    caplog.set_level("ERROR")
    assert _cache_key(_fake_jwt("gid:AAAA123")) == "ikg_schema:gid:AAAA123"
    assert _cache_key("") == "ikg_schema:unconfigured"

    with pytest.raises(InvalidAppAgentKeyError):
        _cache_key("not-a-jwt")
    with pytest.raises(InvalidAppAgentKeyError):
        _cache_key(jwt.encode({"no_sub_here": "x"}, key="", algorithm="none"))

    assert "not a well-formed JWT" in caplog.text
    assert "missing a string 'sub' claim" in caplog.text


def test_schema_does_not_collide_across_different_environments(mocker):
    """SchemaClients for different environments sharing a cache region don't see each other's schema."""
    shared_cache = _memory_cache()
    other_schema = {"graph": {"nodes": {"Other": {}}, "edges": []}}

    _mock_sync_client(mocker, response=_mock_response(mocker))
    client_a = SchemaClient("https://a.example.com", _fake_jwt("gid-a"), cache=shared_cache)
    client_a.preload()

    _mock_sync_client(mocker, response=_mock_response(mocker, schema=other_schema))
    client_b = SchemaClient("https://b.example.com", _fake_jwt("gid-b"), cache=shared_cache)
    client_b.preload()

    assert client_a.get_cached() == _SCHEMA
    assert client_b.get_cached() == other_schema


def test_get_cached_fetches_on_demand_when_not_already_cached(mocker, caplog):
    """On a miss get_cached() fetches itself, logging a cache miss, and returns None if unconfigured."""
    caplog.set_level("DEBUG")
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())
    assert client.get_cached() == _SCHEMA
    assert f"cache_key={client._cache_key} hit=False" in caplog.text

    unconfigured_client = SchemaClient("", "", cache=_memory_cache())
    assert unconfigured_client.get_cached() is None


def test_get_cached_treats_a_cache_backend_failure_as_schema_unavailable(mocker):
    """A cache backend error is treated as an unavailable schema, not raised."""
    cache = mocker.Mock(get=mocker.Mock(side_effect=ConnectionError("redis unreachable")))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached() is None


def test_peek_cached_never_fetches(mocker):
    """peek_cached() answers from the cache only and returns None on a miss."""
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    cache = _memory_cache()
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.peek_cached() is None
    client_mock.get.assert_not_called()

    client.preload()
    assert client.peek_cached() == _SCHEMA
    client_mock.get.assert_called_once()


def test_get_cached_refreshes_on_demand_when_entry_expired(mocker):
    """An expired entry is served stale while a background refresh re-fetches it."""
    cache, executor = _memory_cache_with_async_refresh()
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)
    client.preload()
    assert client_mock.get.call_count == 1

    _expire(cache)

    assert client.get_cached() == _SCHEMA
    _wait_for_async_refresh(executor)
    assert client_mock.get.call_count == _EXPECTED_FETCH_COUNT_AFTER_REFRESH


def test_get_cached_falls_back_to_stale_schema_when_refresh_fails(mocker, caplog):
    """A failed refresh keeps serving the stale schema and logs schema_refresh_failed."""
    caplog.set_level("ERROR")
    cache, executor = _memory_cache_with_async_refresh()
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)
    client.preload()
    _expire(cache)

    client_mock.get.side_effect = httpx2.HTTPError("platform unreachable")

    assert client.get_cached() == _SCHEMA
    _wait_for_async_refresh(executor)
    assert any("schema_refresh_failed" in record.getMessage() for record in caplog.records)


def test_async_refresh_logs_a_malformed_response_instead_of_swallowing_it(mocker, caplog):
    """A malformed response during a background refresh is logged, not silently lost."""
    caplog.set_level("ERROR")
    cache, executor = _memory_cache_with_async_refresh()
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)
    client.preload()
    _expire(cache)

    client_mock.get.return_value = _mock_malformed_json_response(mocker)

    assert client.get_cached() == _SCHEMA
    _wait_for_async_refresh(executor)
    assert any("schema_refresh_failed" in record.getMessage() for record in caplog.records)
    assert any("Failed to load data schema" in record.getMessage() for record in caplog.records)


def test_async_refresh_logs_when_the_cache_write_itself_fails(mocker, caplog):
    """A cache write failure during a background refresh is logged, not silently lost."""
    caplog.set_level("ERROR")
    cache, executor = _memory_cache_with_async_refresh()
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)
    client.preload()
    _expire(cache)

    mocker.patch.object(cache, "set", side_effect=ConnectionError("redis write failed"))

    assert client.get_cached() == _SCHEMA
    _wait_for_async_refresh(executor)
    assert any("schema_refresh_failed" in record.getMessage() for record in caplog.records)
