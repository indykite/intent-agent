"""Tests for SchemaClient: fetching, caching, and TTL-driven on-demand refresh."""

import json

import httpx2
import jwt
import pytest
from dogpile.cache.region import make_region

from indykite.intent_agent.errors import InvalidAppAgentKeyError
from indykite.intent_agent.schema import SchemaClient, _cache_key

_SCHEMA = {"graph": {"nodes": {}, "edges": []}}
_EXPECTED_FETCH_COUNT_AFTER_REFRESH = 2


def _fake_jwt(sub):
    """Build a real (unsigned, alg=none) JWT with the given `sub` claim, via the same library
    _jwt_claim decodes with - hand-rolling the encoding here instead would risk drifting from
    what jwt.decode actually requires, the same class of bug this replaced.
    """
    return jwt.encode({"sub": sub}, key="", algorithm="none")


# A real JWT for tests that don't care about the specific sub claim, just that it's valid -
# SchemaClient now requires a well-formed JWT for any non-blank app_agent_key.
_APP_AGENT_KEY = _fake_jwt("test-app-agent")


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


def _mock_response(mocker, schema=_SCHEMA):
    """Build a fake httpx2 response returning schema from .json(), never raising on
    .raise_for_status(). status_code/text are real values (not further Mocks) since load()
    logs both verbatim, including a slice of .text - a bare Mock isn't subscriptable.
    """
    response = mocker.Mock(json=mocker.Mock(return_value=schema))
    response.status_code = 200
    response.text = json.dumps(schema)
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
    """An HTTP error from the platform must return None, not raise - and the two very
    different underlying causes (network failure vs. a rejected credential) must be
    distinguishable in the logs, not collapsed into one generic message.
    """
    caplog.set_level("ERROR")
    _mock_sync_client(mocker, side_effect=error)
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())
    assert client.load() is None
    assert str(error) in caplog.text


def test_load_returns_schema_on_success(mocker):
    """A successful fetch returns the parsed JSON schema."""
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())
    assert client.load() == _SCHEMA


def test_load_logs_a_truncated_preview_of_the_response_body(mocker, caplog):
    """load() must log the response status and a body[:500] preview at DEBUG level -
    diagnoses an unexpected/empty schema or a non-JSON error page directly from the logs,
    without guessing from the parsed summary alone. A body longer than 500 characters verifies
    the truncation actually happens, not just that some text got logged.
    """
    caplog.set_level("DEBUG")
    large_schema = {"graph": {"nodes": {"Padding": {"description": "x" * 600}}, "edges": []}}
    _mock_sync_client(mocker, response=_mock_response(mocker, schema=large_schema))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())

    assert client.load() == large_schema

    full_body = json.dumps(large_schema)
    assert len(full_body) > len(full_body[:500])
    assert f"Schema response: status=200 body[:500]={full_body[:500]!r}" in caplog.text
    # The full (untruncated) body can't be a substring of a message that only ever logged its
    # first 500 characters - a stronger check than looking for a specific missing fragment,
    # which a uniform-character payload could satisfy by coincidence.
    assert full_body not in caplog.text


@pytest.mark.parametrize("backend", ["memory", "redis"])
def test_preload_and_get_cached_round_trip(mocker, caplog, backend):
    """preload() then get_cached() must return the same schema, on both the default in-process
    Memory cache and a Redis-backed CacheRegion (the real dogpile Redis backend - only its
    network client is faked, so this exercises the actual serialize/deserialize path). Must
    also log the cache key and hit/miss flag at debug level for troubleshooting.
    """
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
    """A well-formed app_agent_key JWT's sub claim (a stable, non-secret GID) must be used
    directly. A blank app_agent_key (the supported "IK not configured" state) must map to a
    stable placeholder instead of attempting to decode. Anything else non-blank must be a
    valid JWT with a sub claim - either failure (not a JWT at all, or missing the claim) is a
    real config error and must raise a clear InvalidAppAgentKeyError, not a raw jwt/TypeError,
    and must not degrade quietly to a hashed key. The two failure modes must be distinguishable
    in the logs, not just via the identical exception message.
    """
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
    """Two SchemaClient instances for different base_url/app_agent_key pairs sharing the same
    cache region must not read/write each other's schema - a single fixed cache key would let
    one silently overwrite the other in a shared Redis backend.
    """
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
    """get_cached() must fetch-and-cache on its own when nothing is cached yet: either because
    preload() was never called (fetch succeeds) or because there's no config to fetch with at
    all (fetch fails, returns None instead of raising). Must also log the cache key and hit/miss
    flag at debug level for troubleshooting.
    """
    caplog.set_level("DEBUG")
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())
    assert client.get_cached() == _SCHEMA
    assert f"cache_key={client._cache_key} hit=False" in caplog.text

    unconfigured_client = SchemaClient("", "", cache=_memory_cache())
    assert unconfigured_client.get_cached() is None


def test_get_cached_refreshes_on_demand_when_entry_expired(mocker):
    """An expired cache entry must trigger a fresh load() and re-cache, not just sit stale."""
    cache = make_region().configure("dogpile.cache.memory", expiration_time=100)
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)
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
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)
    client.preload()
    _expire(cache)

    client_mock.get.side_effect = httpx2.HTTPError("platform unreachable")

    assert client.get_cached() == _SCHEMA
    assert any("schema_refresh_failed" in record.getMessage() for record in caplog.records)
