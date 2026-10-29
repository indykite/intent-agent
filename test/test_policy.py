"""Tests for PolicyClient: fetching, per-subject_type caching, and retry-then-give-up."""

import json
import threading
import time
from http import HTTPStatus

import httpx2
import jwt
import pytest
from dogpile.cache.api import NO_VALUE
from dogpile.cache.backends.redis import RedisBackend
from dogpile.cache.region import make_region

from indykite.intent_agent.dependency_status import IKG, REDIS, DependencyStatus
from indykite.intent_agent.policy import (
    _FAILURE_CACHE_SECONDS,
    PolicyClient,
    _cache_key,
    _FailedFetch,
    _failure_backoff_seconds,
)

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


def _fake_jwt(sub):
    """An unsigned JWT carrying `sub`."""
    return jwt.encode({"sub": sub}, key="", algorithm="none")


_APP_AGENT_KEY = _fake_jwt("test-app-agent")
_SUBJECT_TYPE = "Person"


def _memory_cache():
    return make_region().configure("dogpile.cache.memory")


def _mock_sync_client(mocker, *, response=None, side_effect=None):
    """Patch httpx2.Client so client.get returns `response` or follows `side_effect` (one exception or a list)."""
    client = mocker.MagicMock()
    client.__enter__ = mocker.Mock(return_value=client)
    client.__exit__ = mocker.Mock(return_value=False)
    client.get = mocker.Mock(return_value=response, side_effect=side_effect)
    mocker.patch("indykite.intent_agent.policy.httpx2.Client", return_value=client)
    return client


def _mock_response(mocker, results=None):
    """A fake 200 response whose .json() is {"results": results}, defaulting to _POLICIES."""
    if results is None:
        results = _POLICIES
    body = {"results": results}
    response = mocker.Mock(json=mocker.Mock(return_value=body))
    response.status_code = 200
    response.text = json.dumps(body)
    response.raise_for_status = mocker.Mock()
    return response


def _mock_malformed_response(mocker, *, json_value=None, json_error=None):
    """A 2xx response whose .json() raises `json_error` or returns a malformed `json_value`."""
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
    """A response whose raise_for_status() raises an httpx2.HTTPStatusError with `status_code`."""
    response = mocker.Mock()
    response.status_code = status_code
    response.text = "error body"
    response.raise_for_status = mocker.Mock(
        side_effect=httpx2.HTTPStatusError("error", request=mocker.Mock(), response=response)
    )
    return response


def _expire(cache):
    """Backdate every cached entry's creation time so dogpile treats it as expired."""
    for key, cached in cache.backend._cache.items():
        expired_ct = time.time() - 1_000_000
        cache.backend._cache[key] = cached._replace(metadata={**cached.metadata, "ct": expired_ct})


@pytest.mark.parametrize(
    ("base_url", "app_agent_key"),
    [
        pytest.param("", _APP_AGENT_KEY, id="blank_base_url"),
        pytest.param("https://ikg.example.com", "", id="blank_app_agent_key"),
    ],
)
def test_get_cached_returns_none_when_config_incomplete(base_url, app_agent_key):
    """Missing base_url or app_agent_key means policies unavailable, no request attempted."""
    client = PolicyClient(base_url, app_agent_key, cache=_memory_cache())
    assert client.get_cached(_SUBJECT_TYPE) == (None, True)


def test_get_cached_returns_policies_on_success(mocker):
    """A successful fetch returns the results list, unwrapped from the {"results": [...]} envelope."""
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())
    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)


def test_get_cached_sends_the_subject_type_query_param(mocker):
    """The request must filter by the caller's subject_type, per the endpoint's contract."""
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())

    client.get_cached(_SUBJECT_TYPE)

    client_mock.get.assert_called_once_with(
        "https://ikg.example.com/access/v1/policies",
        headers={"X-IK-ClientKey": _APP_AGENT_KEY},
        params={"subject_type": _SUBJECT_TYPE},
    )


def test_get_cached_logs_the_request_url_and_a_truncated_response_preview(mocker, caplog):
    """The request URL and a truncated response body preview are logged at DEBUG."""
    caplog.set_level("DEBUG")
    large_policies = [{"policy": {"description": "x" * 600}, "tags": []}]
    _mock_sync_client(mocker, response=_mock_response(mocker, results=large_policies))
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())

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
    """One failed attempt for `kind`: an exception for client.get to raise, or a malformed 2xx response."""
    if kind == "http_error":
        return httpx2.ConnectTimeout("connect timed out")
    if kind == "invalid_json":
        return _mock_malformed_response(mocker, json_error=json.JSONDecodeError("bad json", "malformed", 0))
    return _mock_malformed_response(mocker, json_value=_MALFORMED_JSON_VALUES[kind])


@pytest.mark.parametrize("kind", ["http_error", "invalid_json", *_MALFORMED_JSON_VALUES])
def test_get_cached_retries_then_succeeds(mocker, kind):
    """A failed attempt (HTTP error or malformed body) is retried and a later success is returned."""
    failure = _one_failure(mocker, kind)
    client_mock = _mock_sync_client(mocker, side_effect=[failure, failure, _mock_response(mocker)])
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3, cache=_memory_cache())

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS


@pytest.mark.parametrize("kind", ["http_error", "invalid_json", *_MALFORMED_JSON_VALUES])
def test_get_cached_returns_none_after_exhausting_retries(mocker, caplog, kind):
    """When every attempt fails, get_cached returns (None, True) and logs a WARNING instead of raising."""
    caplog.set_level("WARNING")
    failure = _one_failure(mocker, kind)
    client_mock = _mock_sync_client(mocker, side_effect=[failure] * _RETRY_ATTEMPTS)
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3, cache=_memory_cache())

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS
    assert "after 3 attempts" in caplog.text


def test_get_cached_truncates_a_malformed_body_in_the_warning_log(mocker, caplog):
    """A malformed body is truncated in the retry-exhausted WARNING log."""
    huge_body_size = 10_000
    truncation_ceiling = 1_000  # comfortably above _ERROR_BODY_PREVIEW_CHARS, well below huge_body_size
    caplog.set_level("WARNING")
    huge_body = {"not_results": "x" * huge_body_size}
    _mock_sync_client(mocker, response=_mock_malformed_response(mocker, json_value=huge_body))
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=1, cache=_memory_cache())

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert "xxxxxxxxxx" in caplog.text
    assert caplog.text.count("x") < truncation_ceiling


def test_get_cached_does_not_retry_a_4xx_response(mocker, caplog):
    """A 4xx can't succeed on retry, so it fails after a single attempt."""
    caplog.set_level("WARNING")
    forbidden = _mock_status_error_response(mocker, 403)
    client_mock = _mock_sync_client(mocker, response=forbidden)
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3, cache=_memory_cache())

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == 1
    assert "after 1 attempt" in caplog.text
    assert "after 1 attempts" not in caplog.text


def test_get_cached_still_retries_a_5xx_response(mocker):
    """A 5xx is retried like a connection error."""
    server_error = _mock_status_error_response(mocker, 503)
    client_mock = _mock_sync_client(mocker, side_effect=[server_error] * _RETRY_ATTEMPTS)
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3, cache=_memory_cache())

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS


@pytest.mark.parametrize("status", [408, 429], ids=["request_timeout", "too_many_requests"])
def test_get_cached_still_retries_408_and_429(mocker, status):
    """408 and 429 are transient, so they are retried like a 5xx."""
    transient = _mock_status_error_response(mocker, status)
    client_mock = _mock_sync_client(mocker, side_effect=[transient] * _RETRY_ATTEMPTS)
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3, cache=_memory_cache())

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS


def test_get_cached_caches_a_failure_briefly(mocker):
    """A failed fetch is cached for a short backoff, so the next call makes no new request."""
    error = httpx2.ConnectTimeout("connect timed out")
    client_mock = _mock_sync_client(mocker, side_effect=error)
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3, cache=_memory_cache())

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS


def test_get_cached_retries_again_once_the_failure_backoff_expires(mocker):
    """Once the failure backoff expires, the next call fetches again."""
    error = httpx2.ConnectTimeout("connect timed out")
    client_mock = _mock_sync_client(mocker, side_effect=[error] * _RETRY_ATTEMPTS + [_mock_response(mocker)])
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3, cache=_memory_cache())
    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS

    mocker.patch("indykite.intent_agent.policy.time.time", return_value=time.time() + _FAILURE_CACHE_SECONDS)

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    assert client_mock.get.call_count == _RETRY_ATTEMPTS + 1


def test_failure_backoff_is_clamped_to_a_shorter_success_ttl():
    """The failure backoff never outlasts the cache region's own expiration time."""
    short_ttl_seconds = 5
    short_ttl_cache = make_region().configure("dogpile.cache.memory", expiration_time=short_ttl_seconds)
    assert _failure_backoff_seconds(short_ttl_cache) == short_ttl_seconds

    long_ttl_seconds = 300
    long_ttl_cache = make_region().configure("dogpile.cache.memory", expiration_time=long_ttl_seconds)
    assert _failure_backoff_seconds(long_ttl_cache) == _FAILURE_CACHE_SECONDS


def test_get_cached_does_not_delete_a_concurrently_recovered_entry(mocker):
    """Evicting an expired failure keeps a fresh value another caller cached in between."""
    # The wrapped .get() writes a fresh value on the re-check right before the eviction.
    cache = _memory_cache()
    cache_key = _cache_key(_APP_AGENT_KEY, _SUBJECT_TYPE)
    cache.set(cache_key, _FailedFetch(expires_at=time.time() - 1, reason="connect timed out"))
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    real_get = cache.get
    calls = {"n": 0}
    recheck_before_delete_call = 2

    def _get_with_concurrent_recovery(key, **kwargs):
        calls["n"] += 1
        if calls["n"] == recheck_before_delete_call:
            cache.set(key, _POLICIES)
        return real_get(key, **kwargs)

    mocker.patch.object(cache, "get", side_effect=_get_with_concurrent_recovery)

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    client_mock.get.assert_not_called()


def test_get_cached_treats_a_cache_backend_failure_as_an_unresolved_fetch(mocker):
    """A cache backend error is treated as a failed fetch, not raised."""
    cache = mocker.Mock(get=mocker.Mock(side_effect=ConnectionError("redis unreachable")))
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached(_SUBJECT_TYPE) == (None, True)


def test_get_cached_coalesces_concurrent_misses_into_one_fetch(mocker):
    """Concurrent callers on the same cache miss share one fetch."""
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
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())

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
    """peek_cached() answers from the cache only, for a hit and a cached failure alike."""
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    cache = _memory_cache()
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.peek_cached(_SUBJECT_TYPE) == (False, None)
    client_mock.get.assert_not_called()

    client.get_cached(_SUBJECT_TYPE)
    assert client.peek_cached(_SUBJECT_TYPE) == (True, _POLICIES)
    assert client_mock.get.call_count == 1

    error = httpx2.ConnectTimeout("connect timed out")
    client_mock.get.side_effect = error
    failing_client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=1, cache=_memory_cache())
    failing_client.get_cached(_SUBJECT_TYPE)
    assert failing_client.peek_cached(_SUBJECT_TYPE) == (True, None)


def test_get_cached_caches_per_subject_type(mocker):
    """Each subject_type is fetched and cached separately."""
    other_policies = [{"policy": {"subject": {"type": "Company"}, "actions": ["CAN_SELL"]}, "tags": []}]
    client_mock = _mock_sync_client(
        mocker, side_effect=[_mock_response(mocker), _mock_response(mocker, results=other_policies)]
    )
    cache = _memory_cache()
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached("Person") == (_POLICIES, False)
    assert client.get_cached("Company") == (other_policies, False)
    assert client_mock.get.call_count == _FETCH_COUNT_PER_SUBJECT_TYPE

    assert client.get_cached("Person") == (_POLICIES, False)
    assert client.get_cached("Company") == (other_policies, False)
    assert client_mock.get.call_count == _FETCH_COUNT_PER_SUBJECT_TYPE


def test_cache_key_scopes_by_environment_and_subject_type():
    """Cache keys differ by both environment (app_agent_key) and subject_type."""
    assert _cache_key(_fake_jwt("gid-a"), "Person") != _cache_key(_fake_jwt("gid-a"), "Company")
    assert _cache_key(_fake_jwt("gid-a"), "Person") != _cache_key(_fake_jwt("gid-b"), "Person")
    assert _cache_key("", "Person") == "ikg_policies:unconfigured:Person"


def test_policies_do_not_collide_across_different_environments(mocker):
    """PolicyClients for different environments sharing a cache region don't see each other's policies."""
    shared_cache = _memory_cache()
    other_policies = [{"policy": {"actions": ["CAN_SELL"]}, "tags": []}]

    _mock_sync_client(mocker, response=_mock_response(mocker))
    client_a = PolicyClient("https://a.example.com", _fake_jwt("gid-a"), cache=shared_cache)
    assert client_a.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)

    _mock_sync_client(mocker, response=_mock_response(mocker, results=other_policies))
    client_b = PolicyClient("https://b.example.com", _fake_jwt("gid-b"), cache=shared_cache)
    assert client_b.get_cached(_SUBJECT_TYPE) == (other_policies, False)

    assert client_a.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)


def test_get_cached_refreshes_on_demand_when_entry_expired(mocker):
    """An expired cache entry must trigger a fresh fetch and re-cache, not just sit stale."""
    cache = make_region().configure("dogpile.cache.memory", expiration_time=100)
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)
    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    assert client_mock.get.call_count == 1

    _expire(cache)

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    assert client_mock.get.call_count == _EXPECTED_FETCH_COUNT_AFTER_REFRESH


def test_get_cached_falls_back_to_stale_policies_when_refresh_fails(mocker, caplog):
    """A failed refresh serves the stale policies but still reports fetch_failed and logs the reason."""
    caplog.set_level("WARNING")
    cache = make_region().configure("dogpile.cache.memory", expiration_time=100)
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=1, cache=cache)
    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, False)
    _expire(cache)

    client_mock.get.side_effect = httpx2.HTTPError("platform unreachable")

    assert client.get_cached(_SUBJECT_TYPE) == (_POLICIES, True)
    assert "platform unreachable" in caplog.text


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        pytest.param(200, "ok", id="ok"),
        pytest.param(503, "error", id="server-error"),
        pytest.param(401, "error", id="auth-rejected"),
        pytest.param(404, None, id="answered-4xx-says-nothing-about-ikg"),
    ],
)
def test_fetch_records_ikg_state(mocker, code, expected):
    status = DependencyStatus()
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=1, cache=_memory_cache())
    client.dependency_status = status
    response = _mock_response(mocker) if code == HTTPStatus.OK else _mock_status_error_response(mocker, code)
    _mock_sync_client(mocker, response=response)

    client.get_cached(_SUBJECT_TYPE)

    assert (status.get(IKG).status if status.get(IKG) else None) == expected


def test_redis_cache_access_records_redis_state(mocker):
    status = DependencyStatus()
    cache = mocker.Mock(actual_backend=mocker.Mock(spec=RedisBackend), get=mocker.Mock(return_value=NO_VALUE))
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)
    client.dependency_status = status

    cache.get_or_create = mocker.Mock(side_effect=ConnectionError)
    assert client.get_cached(_SUBJECT_TYPE) == (None, True)
    assert (status.get(REDIS).status, status.get(REDIS).detail) == ("error", "ConnectionError")

    cache.get_or_create = mocker.Mock(return_value=[])
    assert client.get_cached(_SUBJECT_TYPE) == ([], False)
    assert status.get(REDIS).status == "ok"

    status.record_error(REDIS, "ConnectionError")
    cache.get = mocker.Mock(return_value=[])
    assert client.get_cached(_SUBJECT_TYPE) == ([], False)
    assert status.get(REDIS).status == "ok"
