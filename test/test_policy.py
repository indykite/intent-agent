"""Tests for PolicyClient: fetching, per-subject_type caching, and retry-then-give-up (ENG-9671)."""

import asyncio
import json
import threading
import time

import httpx2
import jwt
import pytest
from fakeredis import FakeAsyncRedis

from indykite.intent_agent.errors import InvalidAppAgentKeyError
from indykite.intent_agent.policy import _FAILURE_CACHE_SECONDS, PolicyClient, _cache_key, _failure_backoff_seconds
from indykite.intent_agent.schema import IkConnection
from indykite.redis.storage import MemoryStorageBackend, RedisStorageBackend, StorageBackend

_POLICIES = [
    {
        "policy": {
            "meta": {"policy_version": "2.0-kbac"},
            "subject": {"type": "Person"},
            "actions": ["CAN_BUY"],
            "resource": {"type": "Car"},
            "condition": {"cypher": "MATCH (subject) MATCH (resource) WHERE true"},
        },
        "tags": [],
    },
]
_EXPECTED_FETCH_COUNT_AFTER_REFRESH = 2
_RETRY_ATTEMPTS = 3
_FETCH_COUNT_PER_SUBJECT_TYPE = 2
_DEFAULT_REFRESH_TTL_SECONDS = 100.0


def _fake_jwt(sub):
    """Same unsigned-JWT helper as test_schema.py, for the same reason: never hand-roll the
    encoding separately from what jwt.decode actually requires.
    """
    return jwt.encode({"sub": sub}, key="", algorithm="none")


_APP_AGENT_KEY = _fake_jwt("test-app-agent")
_SUBJECT_TYPE = "Person"


def _memory_backend() -> MemoryStorageBackend:
    return MemoryStorageBackend()


def _redis_backend() -> RedisStorageBackend:
    return RedisStorageBackend(FakeAsyncRedis())


def _client(base_url: str, app_agent_key: str, *, cache: StorageBackend | None = None, **kwargs) -> PolicyClient:
    """Build a PolicyClient with a sensible default cache/TTL, for tests that don't care."""
    return PolicyClient(
        IkConnection(base_url, app_agent_key),
        cache=cache if cache is not None else _memory_backend(),
        refresh_ttl_seconds=kwargs.pop("refresh_ttl_seconds", _DEFAULT_REFRESH_TTL_SECONDS),
        **kwargs,
    )


def _mock_sync_client(mocker, *, response=None, side_effect=None):
    """Patch httpx2.Client so `with httpx2.Client(...) as client: client.get(...)` yields/raises
    as given - `side_effect` may be a single exception (every call raises) or a list (one
    outcome per call, e.g. [error, error, response] to exercise a retry that succeeds late).
    """
    client = mocker.MagicMock()
    client.__enter__ = mocker.Mock(return_value=client)
    client.__exit__ = mocker.Mock(return_value=False)
    client.get = mocker.Mock(return_value=response, side_effect=side_effect)
    mocker.patch("indykite.intent_agent.policy.httpx2.Client", return_value=client)
    return client


def _mock_response(mocker, results=None):
    """Build a fake httpx2 response returning {"results": results} from .json(), never raising
    on .raise_for_status(). status_code/text are real values since _load() logs both verbatim.

    Defaults to None rather than the _POLICIES list directly (review: mutable default) -
    _load() returns this same list object back to its caller with no copy, so aliasing the
    shared module-level constant into it would let any test that later mutates a get_cached()
    result corrupt _POLICIES for every other test in the file.
    """
    if results is None:
        results = _POLICIES
    body = {"results": results}
    response = mocker.Mock(json=mocker.Mock(return_value=body))
    response.status_code = 200
    response.text = json.dumps(body)
    response.raise_for_status = mocker.Mock()
    return response


def _mock_malformed_response(mocker, *, json_value=None, json_error=None):
    """A 2xx response whose body is not the documented {"results": [...]} envelope - either
    `.json()` itself raises (invalid JSON) or it returns something structurally wrong
    (`json_value`, e.g. not a dict, or missing/non-list "results").
    """
    response = mocker.Mock()
    response.status_code = 200
    response.text = "malformed" if json_error else json.dumps(json_value)
    response.raise_for_status = mocker.Mock()
    if json_error is not None:
        response.json = mocker.Mock(side_effect=json_error)
    else:
        response.json = mocker.Mock(return_value=json_value)
    return response


def _mock_status_error_response(mocker, status_code):
    """A response whose raise_for_status() raises a real httpx2.HTTPStatusError carrying that
    status_code - for testing the 4xx-vs-5xx retry distinction (review).
    """
    response = mocker.Mock()
    response.status_code = status_code
    response.text = "error body"
    response.raise_for_status = mocker.Mock(
        side_effect=httpx2.HTTPStatusError("error", request=mocker.Mock(), response=response)
    )
    return response


def _backdate_cached_at(cache: StorageBackend, key: str, *, by_seconds: float) -> None:
    """Same deterministic-expiry trick as test_schema.py's - directly rewrite the cached
    envelope's cached_at earlier by by_seconds, rather than sleeping past refresh_ttl_seconds.
    """
    import asyncio

    async def _run():
        encoded = await cache.get(key)
        envelope = json.loads(encoded)
        envelope["cached_at"] -= by_seconds
        await cache.set(key, json.dumps(envelope).encode("utf-8"))

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("base_url", "app_agent_key"),
    [
        pytest.param("", _APP_AGENT_KEY, id="blank_base_url"),
        pytest.param("https://ikg.example.com", "", id="blank_app_agent_key"),
    ],
)
def test_get_cached_returns_none_when_config_incomplete(base_url, app_agent_key):
    """Missing base_url or app_agent_key means policies unavailable, no request attempted."""
    client = _client(base_url, app_agent_key)
    assert client.get_cached(_SUBJECT_TYPE) == (None, True)


def test_get_cached_returns_policies_on_success(mocker):
    """A successful fetch returns the results list, unwrapped from the {"results": [...]} envelope."""
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY)
    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)


@pytest.mark.parametrize("backend", ["memory", "redis"])
def test_get_cached_round_trips_on_both_backends(mocker, backend):
    """The success envelope's encode/decode path must work identically on the in-process
    Memory backend and a Redis-backed one (the real redis.asyncio client, faked at the network
    level with fakeredis) - a second call must be a cache hit either way, no new fetch.
    """
    cache = _redis_backend() if backend == "redis" else _memory_backend()
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    assert client_mock.get.call_count == 1


def test_get_cached_sends_the_subject_type_query_param(mocker):
    """The request must filter by the caller's subject_type, per the endpoint's contract."""
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY)

    client.get_cached(_SUBJECT_TYPE)

    client_mock.get.assert_called_once_with(
        "https://ikg.example.com/access/v1/policies",
        headers={"X-IK-ClientKey": _APP_AGENT_KEY},
        params={"subject_type": _SUBJECT_TYPE},
    )


def test_get_cached_logs_the_request_url_and_a_truncated_response_preview(mocker, caplog):
    """Same diagnostic logging convention as SchemaClient: request URL and a truncated response
    body preview at DEBUG, so an unexpected/empty policy set can be read straight from the logs.
    """
    caplog.set_level("DEBUG")
    large_policies = [{"policy": {"description": "x" * 600}, "tags": []}]
    _mock_sync_client(mocker, response=_mock_response(mocker, results=large_policies))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY)

    assert client.get_cached(_SUBJECT_TYPE) == (large_policies, False)

    assert "Requesting policies from https://ikg.example.com/access/v1/policies" in caplog.text
    full_body = json.dumps({"results": large_policies})
    assert f"Policy response: status=200 body[:500]={full_body[:500]!r}" in caplog.text
    assert full_body not in caplog.text


_MALFORMED_JSON_VALUES = {
    "body_not_a_dict": ["not", "an", "object"],
    "results_missing": {"no_results_key": True},
    "results_not_a_list": {"results": "not-a-list"},
    "soft_error_envelope": {"error": "insufficient API access level"},
}


def _one_failure(mocker, kind):
    """Build one failed attempt's worth of mock state for `kind` - either an exception (raised
    directly by client.get) or a malformed-but-2xx response object (review: invalid JSON, or a
    structurally wrong envelope), covering both ways an attempt can fail.
    """
    if kind == "http_error":
        return httpx2.ConnectTimeout("connect timed out")
    if kind == "invalid_json":
        return _mock_malformed_response(mocker, json_error=json.JSONDecodeError("bad json", "malformed", 0))
    return _mock_malformed_response(mocker, json_value=_MALFORMED_JSON_VALUES[kind])


@pytest.mark.parametrize("kind", ["http_error", "invalid_json", *_MALFORMED_JSON_VALUES])
def test_get_cached_retries_then_succeeds(mocker, kind):
    """A transient failure on an earlier attempt - an HTTP error or a malformed 2xx body alike
    (review) - must not fail the whole fetch; only exhausting every retry does.
    """
    failure = _one_failure(mocker, kind)
    client_mock = _mock_sync_client(mocker, side_effect=[failure, failure, _mock_response(mocker)])
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3)

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS


@pytest.mark.parametrize("kind", ["http_error", "invalid_json", *_MALFORMED_JSON_VALUES])
def test_get_cached_returns_none_after_exhausting_retries(mocker, caplog, kind):
    """Every attempt failing - whether an HTTP error or a malformed 2xx body (review: invalid
    JSON, a non-object body, a missing/non-list "results", or a 2xx "soft error" envelope that
    would otherwise silently become a successful empty policy set and hide a real problem from
    the caller) - must return None (extraction proceeds without policy context, see
    subject_binding.py) rather than raise, and log at WARNING once retries are exhausted, not
    ERROR: a policies outage is never fatal here.
    """
    caplog.set_level("WARNING")
    failure = _one_failure(mocker, kind)
    client_mock = _mock_sync_client(mocker, side_effect=[failure] * _RETRY_ATTEMPTS)
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3)

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS
    assert "after 3 attempts" in caplog.text


def test_get_cached_truncates_a_malformed_body_in_the_warning_log(mocker, caplog):
    """A malformed 2xx response's body must not reach the retry-exhausted WARNING log
    unbounded (review) - that level is always on, unlike the DEBUG-level per-attempt response
    preview, which already truncates the same way; a large or sensitive body must not bypass
    that just because it came from _parse_results's own error message instead.
    """
    huge_body_size = 10_000
    truncation_ceiling = 1_000  # comfortably above _ERROR_BODY_PREVIEW_CHARS, well below huge_body_size
    caplog.set_level("WARNING")
    huge_body = {"not_results": "x" * huge_body_size}
    _mock_sync_client(mocker, response=_mock_malformed_response(mocker, json_value=huge_body))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=1)

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert "xxxxxxxxxx" in caplog.text
    assert caplog.text.count("x") < truncation_ceiling


def test_get_cached_does_not_retry_a_4xx_response(mocker, caplog):
    """A 4xx (review: e.g. the AppAgent lacks ReadAuthZConfigs, or a plain 404) will not
    succeed on retry, so it must fail fast after a single attempt instead of spending the same
    retry budget a transient 5xx/connection error gets.
    """
    caplog.set_level("WARNING")
    forbidden = _mock_status_error_response(mocker, 403)
    client_mock = _mock_sync_client(mocker, response=forbidden)
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3)

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == 1
    assert "after 1 attempt" in caplog.text
    assert "after 1 attempts" not in caplog.text


def test_get_cached_still_retries_a_5xx_response(mocker):
    """Unlike a 4xx, a 5xx is the platform's own problem and might clear up by the next
    attempt, so it keeps the same retry tolerance as a connection error.
    """
    server_error = _mock_status_error_response(mocker, 503)
    client_mock = _mock_sync_client(mocker, side_effect=[server_error] * _RETRY_ATTEMPTS)
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3)

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS


@pytest.mark.parametrize("status", [408, 429], ids=["request_timeout", "too_many_requests"])
def test_get_cached_still_retries_408_and_429(mocker, status):
    """408 (the platform itself timed out on that attempt) and 429 (rate-limited) both describe
    a condition a subsequent attempt can plausibly clear (review) - unlike e.g. a 401/403/404,
    identical retries are not doomed to fail the same way, so these get the same retry
    tolerance as a 5xx instead of failing fast after a single attempt.
    """
    transient = _mock_status_error_response(mocker, status)
    client_mock = _mock_sync_client(mocker, side_effect=[transient] * _RETRY_ATTEMPTS)
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3)

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS


def test_get_cached_caches_a_failure_briefly(mocker):
    """A failed fetch is itself cached for a short backoff (review) - without this, every
    request during an outage would pay the full retry_attempts x timeout_seconds cost again.
    A second call within that window must return None without attempting any new request.
    """
    error = httpx2.ConnectTimeout("connect timed out")
    client_mock = _mock_sync_client(mocker, side_effect=error)
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3)

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS  # no new attempts - backoff still active


def test_get_cached_retries_again_once_the_failure_backoff_expires(mocker):
    """The cached-failure backoff is short and must itself expire - a platform that recovers
    should be retried again soon, not left "marked down" for the full success TTL.
    """
    error = httpx2.ConnectTimeout("connect timed out")
    client_mock = _mock_sync_client(mocker, side_effect=[error] * _RETRY_ATTEMPTS + [_mock_response(mocker)])
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3)
    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS

    mocker.patch("indykite.intent_agent.policy.time.time", return_value=time.time() + _FAILURE_CACHE_SECONDS)

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS + 1


def test_failure_backoff_is_clamped_to_a_shorter_success_ttl():
    """A POLICY_REFRESH_TTL_MINUTES configured under 30 seconds must not let a cached failure
    outlive what the success path's own freshness check would already consider stale -
    otherwise peek_cached() would report a miss before the failure's own backoff elapses,
    defeating the point of caching it at all in that configuration.
    """
    short_ttl_seconds = 5.0
    long_ttl_seconds = 300.0
    assert _failure_backoff_seconds(short_ttl_seconds) == short_ttl_seconds
    assert _failure_backoff_seconds(long_ttl_seconds) == _FAILURE_CACHE_SECONDS


def test_get_cached_treats_a_cache_backend_failure_as_an_unresolved_fetch(mocker):
    """A cache backend failure (e.g. CACHE_BACKEND=redis genuinely unreachable, not just a
    miss) must be treated the same as a policy fetch failure rather than escape as an
    unhandled exception (review) - run_extraction answers policy_unavailable_error either way.
    """
    cache = mocker.Mock(get=mocker.Mock(side_effect=ConnectionError("redis unreachable")))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)


def test_get_cached_treats_a_legacy_or_corrupted_cached_value_as_a_miss(mocker):
    """A stored value that fails to decode into this envelope shape at all (e.g. a value from
    a different cache format under the same key) must be treated as a miss and self-heal via
    the ordinary cache-miss -> fetch -> _store_success() path, not get stuck returning
    (None, True) forever.
    """
    cache = _memory_backend()
    asyncio.run(
        cache.set(
            _cache_key(_APP_AGENT_KEY, _SUBJECT_TYPE),
            b"\x80\x04not valid json, e.g. a leftover pickled value from a retired cache format",
        )
    )
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    # The bad key must actually have been overwritten, not just papered over for this one
    # call - a second, independent read confirms the cache is healed, not still corrupted.
    assert client.peek_cached(_SUBJECT_TYPE) == (True, _POLICIES)


def test_get_cached_treats_a_success_envelope_missing_policies_as_a_miss(mocker):
    """A "success" envelope with no policies list (valid kind, invalid payload - e.g. hand
    corruption, or a future incompatible cache-format change) must self-heal via a fresh fetch,
    not be accepted as a resolved hit - which would otherwise return (None, True) forever until
    the TTL expires, misreporting a healthy fetch as a failure.
    """
    cache = _memory_backend()
    asyncio.run(
        cache.set(
            _cache_key(_APP_AGENT_KEY, _SUBJECT_TYPE),
            json.dumps({"kind": "success", "cached_at": time.time()}).encode(),
        )
    )
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)


def test_get_cached_treats_a_failure_envelope_missing_reason_as_a_miss(mocker):
    """A "failure" envelope with no reason string (valid kind, invalid payload) must likewise
    self-heal via a fresh fetch rather than be accepted as a resolved (still-backing-off)
    failure.
    """
    cache = _memory_backend()
    asyncio.run(
        cache.set(
            _cache_key(_APP_AGENT_KEY, _SUBJECT_TYPE),
            json.dumps({"kind": "failure", "cached_at": time.time()}).encode(),
        )
    )
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)


def test_get_cached_treats_a_non_numeric_cached_at_as_a_miss(mocker):
    """An envelope whose cached_at isn't numeric (e.g. null, from hand corruption or a future
    incompatible cache-format change) must self-heal via a fresh fetch, not be accepted as a
    resolved hit - which would otherwise raise once _resolve() tries `time.time() - cached_at`,
    misreporting a cache-corruption issue as a backend failure forever, since the bad value is
    never replaced.
    """
    cache = _memory_backend()
    asyncio.run(
        cache.set(
            _cache_key(_APP_AGENT_KEY, _SUBJECT_TYPE),
            json.dumps({"kind": "success", "cached_at": None, "policies": []}).encode(),
        )
    )
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)


def test_get_cached_coalesces_concurrent_misses_into_one_fetch(mocker):
    """Concurrent callers observing the same cache miss must coalesce into a single _load()
    behind RefreshGate's per-key lock instead of each firing their own request. This
    is a real concurrency property - proven with actual threads against a real (in-memory)
    backend, not by mocking the lock away.
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
    client = _client("https://ikg.example.com", _APP_AGENT_KEY)

    results = []

    def _call():
        results.append(client.get_cached(_SUBJECT_TYPE))

    threads = [threading.Thread(target=_call) for _ in range(5)]
    for thread in threads:
        thread.start()
    assert load_started.wait(timeout=5), "no thread ever reached _load()"
    release_load.set()
    for thread in threads:
        thread.join(timeout=5)

    assert call_count == 1
    assert results == [(_POLICIES, False)] * len(threads)


def test_peek_cached_never_fetches(mocker):
    """peek_cached() must resolve purely from the cache, without ever calling _load() - a
    caller (review, ENG-9671: run_extraction) uses this to skip the dedicated fetch executor
    entirely once the cache already has an answer, whether that answer is a real hit or a
    still-fresh cached failure's backoff.
    """
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    cache = _memory_backend()
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.peek_cached(_SUBJECT_TYPE) == (False, None)
    client_mock.get.assert_not_called()

    client.get_cached(_SUBJECT_TYPE)
    assert client.peek_cached(_SUBJECT_TYPE) == (True, _POLICIES)
    assert client_mock.get.call_count == 1

    error = httpx2.ConnectTimeout("connect timed out")
    client_mock.get.side_effect = error
    failing_client = _client("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=1)
    failing_client.get_cached(_SUBJECT_TYPE)
    assert failing_client.peek_cached(_SUBJECT_TYPE) == (True, None)


def test_get_cached_caches_per_subject_type(mocker):
    """Two different subject_types must each trigger their own fetch and be cached separately -
    a shared cache entry would leak one subject_type's policies into another's request.
    """
    other_policies = [{"policy": {"subject": {"type": "Company"}, "actions": ["CAN_SELL"]}, "tags": []}]
    client_mock = _mock_sync_client(
        mocker, side_effect=[_mock_response(mocker), _mock_response(mocker, results=other_policies)]
    )
    cache = _memory_backend()
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached("Person") == (_POLICIES, False)
    assert client.get_cached("Company") == (other_policies, False)
    assert client_mock.get.call_count == _FETCH_COUNT_PER_SUBJECT_TYPE

    # Both now cache hits - no further fetches for either subject_type.
    assert client.get_cached("Person") == (_POLICIES, False)
    assert client.get_cached("Company") == (other_policies, False)
    assert client_mock.get.call_count == _FETCH_COUNT_PER_SUBJECT_TYPE


def test_cache_key_scopes_by_environment_and_subject_type():
    """Cache keys must differ across both environment (app_agent_key) and subject_type - either
    axis colliding would let one request's policies leak into an unrelated one's.
    """
    assert _cache_key(_fake_jwt("gid-a"), "Person") != _cache_key(_fake_jwt("gid-a"), "Company")
    assert _cache_key(_fake_jwt("gid-a"), "Person") != _cache_key(_fake_jwt("gid-b"), "Person")
    assert _cache_key("", "Person") == "ikg_policies:unconfigured:Person"


def test_policies_do_not_collide_across_different_environments(mocker):
    """Two PolicyClient instances for different base_url/app_agent_key pairs, same subject_type,
    sharing a cache backend must not read/write each other's policies.
    """
    shared_cache = _memory_backend()
    other_policies = [{"policy": {"actions": ["CAN_SELL"]}, "tags": []}]

    _mock_sync_client(mocker, response=_mock_response(mocker))
    client_a = _client("https://a.example.com", _fake_jwt("gid-a"), cache=shared_cache)
    assert client_a.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)

    _mock_sync_client(mocker, response=_mock_response(mocker, results=other_policies))
    client_b = _client("https://b.example.com", _fake_jwt("gid-b"), cache=shared_cache)
    assert client_b.get_cached(_SUBJECT_TYPE) == (other_policies, False)

    assert client_a.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)


def test_get_cached_refreshes_on_demand_when_entry_expired(mocker):
    """An expired cache entry must trigger a fresh fetch and re-cache, not just sit stale -
    synchronously (unlike SchemaClient, PolicyClient has no background-refresh path), so this
    call itself performs the second fetch and returns its result directly.
    """
    cache = _memory_backend()
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client("https://ikg.example.com", _APP_AGENT_KEY, cache=cache, refresh_ttl_seconds=100.0)
    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    assert client_mock.get.call_count == 1

    _backdate_cached_at(cache, _cache_key(_APP_AGENT_KEY, _SUBJECT_TYPE), by_seconds=101.0)

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    assert client_mock.get.call_count == _EXPECTED_FETCH_COUNT_AFTER_REFRESH


def test_get_cached_falls_back_to_stale_policies_when_refresh_fails(mocker, caplog):
    """Per the same TDD 4.4 principle schema.py already follows (review): a refresh that fails
    for an entry with a still-cached, only-just-expired value must keep serving it rather than
    discarding it for a failure marker - the request currently in flight, and every other one
    during the platform outage, still get a genuinely usable policy set instead of being refused
    (policy_unavailable_error) for no reason. The specific failure reason must still be logged.

    fetch_failed must be True here even though policies is the real, non-None stale list
    (review, High) - this call's own live fetch genuinely failed, so the caller-facing marker
    and the audit trail must say so, not silently read as a healthy fetch just because a
    usable fallback existed.
    """
    caplog.set_level("WARNING")
    cache = _memory_backend()
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = _client(
        "https://ikg.example.com", _APP_AGENT_KEY, cache=cache, retry_attempts=1, refresh_ttl_seconds=100.0
    )
    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    _backdate_cached_at(cache, _cache_key(_APP_AGENT_KEY, _SUBJECT_TYPE), by_seconds=101.0)

    client_mock.get.side_effect = httpx2.HTTPError("platform unreachable")

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, True)
    assert "platform unreachable" in caplog.text


def test_invalid_app_agent_key_is_treated_as_policies_unavailable(caplog):
    """A malformed (non-blank) IK_APP_AGENT_KEY must not raise out of get_cached()/peek_cached()
    - the process already fails loudly at startup in this case elsewhere (see
    InvalidAppAgentKeyError's own docstring); a per-request caller here just needs a clean
    "unavailable" answer.
    """
    caplog.set_level("WARNING")
    client = _client("https://ikg.example.com", "not-a-jwt")

    assert client.peek_cached(_SUBJECT_TYPE) == (True, None)
    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    with pytest.raises(InvalidAppAgentKeyError):
        _cache_key("not-a-jwt", _SUBJECT_TYPE)
