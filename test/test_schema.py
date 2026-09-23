"""Tests for SchemaClient: fetching, caching, and TTL-driven on-demand refresh."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx2
import jwt
import pytest
from fakeredis import FakeAsyncRedis

from indykite.intent_agent.errors import InvalidAppAgentKeyError
from indykite.intent_agent.schema import IkConnection, SchemaClient, _cache_key
from indykite.redis.storage import MemoryStorageBackend, RedisStorageBackend, StorageBackend

_SCHEMA = {"graph": {"nodes": {}, "edges": []}}
_EXPECTED_FETCH_COUNT_AFTER_REFRESH = 2
_DEFAULT_REFRESH_TTL_SECONDS = 100.0


def _fake_jwt(sub):
    """Build a real (unsigned, alg=none) JWT with the given `sub` claim, via the same library
    app_agent_key_gid decodes with - hand-rolling the encoding here instead would risk drifting
    from what jwt.decode actually requires, the same class of bug this replaced.
    """
    return jwt.encode({"sub": sub}, key="", algorithm="none")


# A real JWT for tests that don't care about the specific sub claim, just that it's valid -
# SchemaClient now requires a well-formed JWT for any non-blank app_agent_key.
_APP_AGENT_KEY = _fake_jwt("test-app-agent")


def _memory_backend() -> MemoryStorageBackend:
    """A fresh in-process StorageBackend for a test's SchemaClient."""
    return MemoryStorageBackend()


def _redis_backend() -> RedisStorageBackend:
    """A fresh RedisStorageBackend wrapping an in-memory fake Redis client - exercises the real
    serialize/deserialize path, only the network client itself is faked, no real Redis server
    needed.
    """
    return RedisStorageBackend(FakeAsyncRedis())


def _client(
    base_url: str,
    app_agent_key: str,
    *,
    cache: StorageBackend | None = None,
    executor: ThreadPoolExecutor | None = None,
    **kwargs,
) -> SchemaClient:
    """Build a SchemaClient with a sensible default cache/TTL/executor, for tests that don't
    care about the specifics of any of the three.
    """
    return SchemaClient(
        IkConnection(base_url, app_agent_key),
        cache=cache if cache is not None else _memory_backend(),
        refresh_ttl_seconds=kwargs.pop("refresh_ttl_seconds", _DEFAULT_REFRESH_TTL_SECONDS),
        executor=executor if executor is not None else ThreadPoolExecutor(max_workers=1),
        **kwargs,
    )


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


def _expire_freshness_marker(cache: StorageBackend, client: SchemaClient) -> None:
    """Delete the freshness marker - a deterministic "make this entry look expired" without
    sleeping past refresh_ttl_seconds; the schema itself is left untouched."""
    asyncio.run(cache.delete(client._fresh_key))


def _wait_for_background_refresh(executor: ThreadPoolExecutor) -> None:
    """Block until every refresh already submitted to `executor` has finished - a stale-entry
    get_cached() call kicks its refresh off via RefreshGate and returns immediately (that's the
    point, see SchemaClient.get_cached's docstring), so asserting on the refresh's own side
    effects (a re-fetch, a failure log) needs to wait for it deterministically instead of
    asserting right after get_cached() returns.
    """
    executor.shutdown(wait=True)


@pytest.mark.parametrize(
    ("base_url", "app_agent_key"),
    [
        pytest.param("", _APP_AGENT_KEY, id="blank_base_url"),
        pytest.param("https://ikg.example.com", "", id="blank_app_agent_key"),
    ],
)
def test_load_returns_none_when_config_incomplete(base_url, app_agent_key):
    """Missing base_url or app_agent_key means schema unavailable, no request attempted."""
    client = _client(base_url, app_agent_key)
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
    client = _client("https://ikg.example.com", _APP_AGENT_KEY)
    assert client.load() is None
    assert str(error) in caplog.text


def test_load_returns_none_on_malformed_json_response(mocker, caplog):
    """A 2xx response whose body is not valid JSON must return None, not raise (review):
    resp.json() itself raises ValueError, not httpx2.HTTPError, and must be treated as a
    failed load the same way an HTTP error is rather than escape uncaught - see load()'s
    docstring for why that matters beyond this direct call.
    """
    caplog.set_level("ERROR")
    _mock_sync_client(mocker, response=_mock_malformed_json_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY)
    assert client.load() is None
    assert "Failed to load data schema" in caplog.text


def test_load_returns_schema_on_success(mocker):
    """A successful fetch returns the parsed JSON schema."""
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY)
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
    client = _client("https://ikg.example.com", _APP_AGENT_KEY)

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
    Memory backend and a Redis-backed one (the real redis.asyncio client, faked at the network
    level with fakeredis - only the network client is faked, so this exercises the actual
    encode/decode path). Must also log the cache key and hit flag at debug level for
    troubleshooting.
    """
    caplog.set_level("DEBUG")
    cache = _redis_backend() if backend == "redis" else _memory_backend()
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    client.preload()

    assert client.get_cached() == _SCHEMA
    assert f"cache_key={client._cache_key} hit=True" in caplog.text


def test_cache_key_uses_jwt_sub_claim_blank_placeholder_or_raises(caplog):
    """A well-formed JWT's sub claim is used directly; a blank key maps to a stable
    placeholder. Anything else non-blank must be a valid JWT with a sub claim, or raise a
    clear InvalidAppAgentKeyError - not degrade quietly to a hashed key.
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
    cache backend must not read/write each other's schema - a single fixed cache key would let
    one silently overwrite the other in a shared Redis backend.
    """
    shared_cache = _memory_backend()
    other_schema = {"graph": {"nodes": {"Other": {}}, "edges": []}}

    _mock_sync_client(mocker, response=_mock_response(mocker))
    client_a = _client("https://a.example.com", _fake_jwt("gid-a"), cache=shared_cache)
    client_a.preload()

    _mock_sync_client(mocker, response=_mock_response(mocker, schema=other_schema))
    client_b = _client("https://b.example.com", _fake_jwt("gid-b"), cache=shared_cache)
    client_b.preload()

    assert client_a.get_cached() == _SCHEMA
    assert client_b.get_cached() == other_schema


def test_get_cached_fetches_on_demand_when_not_already_cached(mocker, caplog):
    """get_cached() must fetch-and-cache on its own when nothing is cached yet: either because
    preload() was never called (fetch succeeds) or because there's no config to fetch with at
    all (fetch fails, returns None instead of raising). Must also log the cache key and
    hit=False at debug level for troubleshooting.
    """
    caplog.set_level("DEBUG")
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY)
    assert client.get_cached() == _SCHEMA
    assert f"cache_key={client._cache_key} hit=False" in caplog.text

    unconfigured_client = _client("", "")
    assert unconfigured_client.get_cached() is None


def test_get_cached_coalesces_concurrent_cold_misses_into_one_fetch(mocker):
    """Concurrent callers observing the same cold (never-cached) miss must coalesce into a
    single load() behind RefreshGate's per-key lock instead of each firing their own request. A
    real concurrency property, proven with actual threads against a real (in-memory) backend,
    not by mocking the lock away.
    """
    load_started = threading.Event()
    release_load = threading.Event()
    call_count = 0
    call_count_lock = threading.Lock()

    def _slow_get(*_args, **_kwargs):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        load_started.set()
        assert release_load.wait(timeout=5), "the winning fetch was never released"
        return _mock_response(mocker)

    _mock_sync_client(mocker, side_effect=_slow_get)
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, executor=ThreadPoolExecutor(max_workers=5))

    results = []

    def _call():
        results.append(client.get_cached())

    threads = [threading.Thread(target=_call) for _ in range(5)]
    for thread in threads:
        thread.start()
    assert load_started.wait(timeout=5), "no thread ever reached load()"
    release_load.set()
    for thread in threads:
        thread.join(timeout=5)

    assert call_count == 1
    assert results == [_SCHEMA] * len(threads)


def test_get_cached_treats_a_cache_backend_failure_as_schema_unavailable(mocker):
    """A cache backend failure (e.g. CACHE_BACKEND=redis genuinely unreachable, not just a
    miss) must be treated the same as an unavailable schema rather than escape as an unhandled
    exception (review) - run_extraction's schema_unavailable_error path already exists to
    answer exactly that.
    """
    cache = mocker.Mock(get=mocker.Mock(side_effect=ConnectionError("redis unreachable")))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached() is None


def test_get_cached_treats_a_legacy_or_corrupted_cached_value_as_a_miss(mocker):
    """A stored value that fails to decode into JSON at all must be treated as a miss and
    self-heal via the ordinary cold-fetch-and-store path, not return None forever."""
    cache = _memory_backend()
    asyncio.run(
        cache.set(
            _cache_key(_APP_AGENT_KEY),
            b"\x80\x04not valid json, e.g. a leftover pickled value from a retired cache format",
        )
    )
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached() == _SCHEMA
    assert client_mock.get.call_count == 1
    # The bad key must actually have been overwritten, not just papered over for this one
    # call - a second, independent read confirms the cache is healed, not still corrupted.
    assert client.peek_cached() == _SCHEMA


def test_store_delegates_freshness_to_the_backends_own_ttl_not_manual_bookkeeping():
    """_store() must rely on the backend's own native TTL for staleness, not manual
    cached_at bookkeeping: the primary key gets no TTL, only the separate marker does."""
    cache = _memory_backend()
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache, refresh_ttl_seconds=42.0)

    client._store(_SCHEMA)

    primary_value, primary_expiry = cache._store[client._cache_key]
    _fresh_value, fresh_expiry = cache._store[client._fresh_key]
    assert json.loads(primary_value) == _SCHEMA
    assert primary_expiry is None
    assert fresh_expiry is not None


def test_get_cached_treats_a_malformed_cached_value_as_a_miss(mocker):
    """A syntactically valid JSON value that isn't an object (e.g. null) must self-heal via
    a fresh fetch, not be accepted as a resolved hit and handed back as-is."""
    cache = _memory_backend()
    asyncio.run(cache.set(_cache_key(_APP_AGENT_KEY), json.dumps(None).encode()))
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached() == _SCHEMA
    assert client_mock.get.call_count == 1


def test_peek_cached_never_fetches(mocker):
    """peek_cached() must return None on a miss without ever calling load() - a caller (review,
    ENG-9671: run_extraction) uses this to skip the dedicated fetch executor entirely on a
    cache hit, so it must never itself trigger the fetch get_cached() would.
    """
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    cache = _memory_backend()
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.peek_cached() is None
    client_mock.get.assert_not_called()

    client.preload()
    assert client.peek_cached() == _SCHEMA
    client_mock.get.assert_called_once()


def test_get_cached_refreshes_in_the_background_when_entry_expired(mocker):
    """An expired-but-present cache entry must trigger a fresh load() and re-cache, not just
    sit stale - but get_cached() itself returns the still-present stale value immediately (see
    SchemaClient.get_cached's docstring) rather than waiting on that refresh, so the second
    load() this proves happened is only observable once the background refresh it kicked off
    (via RefreshGate) has actually finished.
    """
    cache = _memory_backend()
    executor = ThreadPoolExecutor(max_workers=1)
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client(
        "https://ikg.example.com", _APP_AGENT_KEY, cache=cache, refresh_ttl_seconds=100.0, executor=executor
    )
    client.preload()
    assert client_mock.get.call_count == 1

    _expire_freshness_marker(cache, client)

    assert client.get_cached() == _SCHEMA
    _wait_for_background_refresh(executor)
    assert client_mock.get.call_count == _EXPECTED_FETCH_COUNT_AFTER_REFRESH


def test_get_cached_falls_back_to_stale_schema_when_refresh_fails(mocker, caplog):
    """Per TDD 4.4: if the background refresh of an expired-but-present entry fails, the
    stale-but-last-known-good schema must still be served (not None) and schema_refresh_failed
    logged once the refresh finishes.
    """
    caplog.set_level("ERROR")
    cache = _memory_backend()
    executor = ThreadPoolExecutor(max_workers=1)
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client(
        "https://ikg.example.com", _APP_AGENT_KEY, cache=cache, refresh_ttl_seconds=100.0, executor=executor
    )
    client.preload()
    _expire_freshness_marker(cache, client)

    client_mock.get.side_effect = httpx2.HTTPError("platform unreachable")

    assert client.get_cached() == _SCHEMA
    _wait_for_background_refresh(executor)
    assert any("schema_refresh_failed" in record.getMessage() for record in caplog.records)


def test_background_refresh_logs_a_malformed_response_instead_of_swallowing_it(mocker, caplog):
    """A 2xx response with a malformed body raises ValueError out of load(), not just
    httpx2.HTTPError - the background refresh must still fall back to the stale schema with
    schema_refresh_failed logged, not vanish silently.
    """
    caplog.set_level("ERROR")
    cache = _memory_backend()
    executor = ThreadPoolExecutor(max_workers=1)
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client(
        "https://ikg.example.com", _APP_AGENT_KEY, cache=cache, refresh_ttl_seconds=100.0, executor=executor
    )
    client.preload()
    _expire_freshness_marker(cache, client)

    client_mock.get.return_value = _mock_malformed_json_response(mocker)

    assert client.get_cached() == _SCHEMA
    _wait_for_background_refresh(executor)
    assert any("schema_refresh_failed" in record.getMessage() for record in caplog.records)
    assert any("Failed to load data schema" in record.getMessage() for record in caplog.records)


class _SetFailsBackend(StorageBackend):
    """Wraps a real StorageBackend but makes every set() raise - simulates a Redis write
    failure (connection drop, OOM, ...) distinct from a load() failure, to prove _store()'s own
    exceptions are handled separately from a failed fetch.
    """

    def __init__(self, inner: StorageBackend) -> None:
        self._inner = inner

    async def get(self, key: str) -> bytes | None:
        return await self._inner.get(key)

    async def set(self, key: str, value: bytes, *, ttl_seconds: float | None = None) -> None:
        msg = "simulated Redis write failure"
        raise ConnectionError(msg)

    async def delete(self, key: str) -> None:
        await self._inner.delete(key)


def test_background_refresh_logs_a_store_failure_instead_of_swallowing_it(mocker, caplog):
    """A cache-backend write failure (e.g. Redis unreachable) during _store() must be caught
    inside the detached background refresh and logged as schema_refresh_failed, the same as a
    load() failure - not escape uncaught and vanish invisibly.
    """
    caplog.set_level("ERROR")
    cache = _memory_backend()
    executor = ThreadPoolExecutor(max_workers=1)
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client(
        "https://ikg.example.com", _APP_AGENT_KEY, cache=cache, refresh_ttl_seconds=100.0, executor=executor
    )
    client.preload()
    _expire_freshness_marker(cache, client)

    client._cache = _SetFailsBackend(cache)

    assert client.get_cached() == _SCHEMA
    _wait_for_background_refresh(executor)
    assert any("schema_refresh_failed" in record.getMessage() for record in caplog.records)


def test_preload_does_not_raise_when_store_fails(mocker, caplog):
    """A _store() write failure during preload() must not propagate: IntentAgent.
    preload_schema() calls preload() with no try/except (app.py's build_app calls
    preload_schema() unguarded too), so an uncaught exception here would crash the whole
    process at startup, contradicting preload_schema's documented "best effort, does not
    block startup" contract.
    """
    caplog.set_level("ERROR")
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=_SetFailsBackend(_memory_backend()))

    client.preload()  # must not raise

    assert any("schema_refresh_failed" in record.getMessage() for record in caplog.records)
    # Nothing was ever cached (every set() fails), so this is a second cold fetch - and must
    # still return the freshly loaded schema rather than None, same as get_cached()'s own
    # cold-fetch path (see test_get_cached_returns_freshly_loaded_schema_even_if_caching_it_fails).
    assert client.get_cached() == _SCHEMA


def test_get_cached_returns_freshly_loaded_schema_even_if_caching_it_fails(mocker, caplog):
    """A cold get_cached() must still return the schema it just loaded, even if _store()
    fails afterward (review: _store()'s two independent cache.set() calls can now partially
    succeed - the primary write landing durably while only the fresh marker write fails -
    and this request must not discard a schema it already has in hand because of that).
    """
    caplog.set_level("ERROR")
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=_SetFailsBackend(_memory_backend()))

    assert client.get_cached() == _SCHEMA
    assert any("Failed to cache freshly loaded schema" in record.getMessage() for record in caplog.records)


def test_get_cached_works_when_called_from_within_a_running_event_loop(mocker):
    """SchemaClient's public API stays synchronous even though StorageBackend's Redis
    implementation is asyncio-only - the actual production path (get_cached() calls happen
    from inside the already-running Starlette/uvicorn event loop, not before it). Running the
    call itself inside asyncio.run() here forces SchemaClient's internal sync/async bridge
    (storage.run_sync) to be invoked from a context that already has its own running loop.
    """
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=_redis_backend())

    async def _call_get_cached_from_inside_a_running_loop():
        return client.get_cached()

    assert asyncio.run(_call_get_cached_from_inside_a_running_loop()) == _SCHEMA
