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


def _mock_response(mocker, schema=None):
    """Build a fake httpx2 response returning schema from .json(), never raising on
    .raise_for_status(). status_code/text are real values (not further Mocks) since load()
    logs both verbatim, including a slice of .text - a bare Mock isn't subscriptable.

    Defaults to None rather than the _SCHEMA dict directly (review, mirrors test_policy.py's
    _mock_response: mutable default) - load() returns this same object back to its caller with
    no copy, so aliasing the shared module-level constant into it would let any test that later
    mutates a get_cached() result corrupt _SCHEMA for every other test in the file.
    """
    if schema is None:
        schema = _SCHEMA
    response = mocker.Mock(json=mocker.Mock(return_value=schema))
    response.status_code = 200
    response.text = json.dumps(schema)
    response.raise_for_status = mocker.Mock()
    return response


def _expire(cache):
    """Force every entry in this Memory-backed CacheRegion to read as expired, deterministically -
    backdates the stored CachedValue's creation time instead of sleeping past expiration_time,
    which would make the test slow and racy against real wall-clock timing.

    Backdates to a real past timestamp rather than zeroing it out: dogpile's own Lock treats a
    creation time of 0 as "no value has ever been created" (Lock._has_value: createdtime > 0),
    not "a value exists but is stale" - zeroing it would make get_or_create block waiting to
    acquire the creation lock synchronously instead of returning the stale value immediately
    and kicking off the async refresh, silently exercising the wrong code path (review).
    """
    for key, cached in cache.backend._cache.items():
        expired_ct = time.time() - 1_000_000
        cache.backend._cache[key] = cached._replace(metadata={**cached.metadata, "ct": expired_ct})


def _memory_cache_with_async_refresh():
    """Build a Memory-backed CacheRegion wired with the real async_creation_runner (review),
    matching how build_schema_cache actually configures it - a test exercising the "expired but
    present" path needs this, or get_cached's return (the immediate stale value) would race the
    background refresh it kicks off. A test asserting on that refresh's own side effects (a
    re-fetch, a failure log) must wait for it deterministically - see _wait_for_async_refresh -
    rather than assert right after get_cached() returns.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    cache = make_region(async_creation_runner=_make_async_creation_runner(executor)).configure(
        "dogpile.cache.memory", expiration_time=100
    )
    return cache, executor


def _wait_for_async_refresh(executor):
    """Block until every refresh already submitted to `executor` has finished - see
    _memory_cache_with_async_refresh.
    """
    executor.shutdown(wait=True)


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


def _mock_malformed_json_response(mocker):
    """A 2xx response whose body is not valid JSON at all - resp.json() itself raises
    json.JSONDecodeError (a ValueError subclass), not httpx2.HTTPError.
    """
    response = mocker.Mock()
    response.status_code = 200
    response.text = "not json"
    response.raise_for_status = mocker.Mock()
    response.json = mocker.Mock(side_effect=json.JSONDecodeError("bad json", "not json", 0))
    return response


def test_load_returns_none_on_malformed_json_response(mocker, caplog):
    """A 2xx response whose body is not valid JSON must return None, not raise (review):
    resp.json() itself raises ValueError, not httpx2.HTTPError, and must be treated as a
    failed load the same way an HTTP error is rather than escape uncaught - see load()'s
    docstring for why that matters beyond this direct call.
    """
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
    """load() must log the request URL, and the response status and a body[:500] preview, at
    DEBUG level - diagnoses an unexpected/empty schema or a non-JSON error page directly from
    the logs, without guessing from the parsed summary alone. A body longer than 500 characters
    verifies the truncation actually happens, not just that some text got logged.
    """
    caplog.set_level("DEBUG")
    large_schema = {"graph": {"nodes": {"Padding": {"description": "x" * 600}}, "edges": []}}
    _mock_sync_client(mocker, response=_mock_response(mocker, schema=large_schema))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())

    assert client.load() == large_schema

    assert "Requesting schema from https://ikg.example.com/data-schema/v1" in caplog.text
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


def test_get_cached_treats_a_cache_backend_failure_as_schema_unavailable(mocker):
    """A cache backend failure (e.g. CACHE_BACKEND=redis genuinely unreachable, not just a
    miss) must be treated the same as an unavailable schema rather than escape as an unhandled
    exception (review) - run_extraction's schema_unavailable_error path already exists to
    answer exactly that.
    """
    cache = mocker.Mock(get=mocker.Mock(side_effect=ConnectionError("redis unreachable")))
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached() is None


def test_peek_cached_never_fetches(mocker):
    """peek_cached() must return None on a miss without ever calling load() - a caller (review,
    ENG-9671: run_extraction) uses this to skip the dedicated fetch executor entirely on a
    cache hit, so it must never itself trigger the fetch get_cached() would.
    """
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    cache = _memory_cache()
    client = SchemaClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.peek_cached() is None
    client_mock.get.assert_not_called()

    client.preload()
    assert client.peek_cached() == _SCHEMA
    client_mock.get.assert_called_once()


def test_get_cached_refreshes_on_demand_when_entry_expired(mocker):
    """An expired cache entry must trigger a fresh load() and re-cache, not just sit stale.

    get_cached() itself returns the still-present stale value immediately (review,
    async_creation_runner - see SchemaClient.get_cached's docstring) rather than waiting on
    that refresh, so the second _load() this proves happened is only observable once the
    background refresh it kicked off has actually finished.
    """
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
    """Per TDD 4.4: if the on-demand refresh fails, the expired-but-last-known-good schema
    must still be served (not None) and schema_refresh_failed logged - a transient platform
    outage must not take down an already-running agent.

    get_cached() returns that stale value immediately, before the (about to fail) refresh
    this test mocks even runs (review, async_creation_runner) - the failure log is only
    observable once that background refresh has actually finished, see
    _memory_cache_with_async_refresh.
    """
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
    """Review finding: a 2xx response with a malformed body raises ValueError out of load(),
    not just httpx2.HTTPError - and unlike a synchronous get_cached() call, an exception
    raised inside the async refresh thread is never surfaced anywhere (its Future is
    fire-and-forget, nobody calls .result() on it), so it would otherwise vanish with nothing
    logged at all instead of falling back to the stale schema with schema_refresh_failed
    logged like every other refresh failure.
    """
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
    """Review finding: a successful fetch's own cache.set() can still raise (e.g. a Redis write
    dropping mid-refresh even though the fetch that just preceded it succeeded), separate from
    creator() failing - the same fire-and-forget reasoning as the malformed-response case above
    means this would otherwise vanish uncaught with nothing logged either.
    """
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
