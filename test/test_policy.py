"""Tests for PolicyClient: fetching, per-subject_type caching, and retry-then-give-up (ENG-9671)."""

import json

import httpx2
import jwt
import pytest
from dogpile.cache.region import make_region

from indykite.intent_agent.policy import PolicyClient, _cache_key

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
    """Same unsigned-JWT helper as test_schema.py, for the same reason: never hand-roll the
    encoding separately from what jwt.decode actually requires.
    """
    return jwt.encode({"sub": sub}, key="", algorithm="none")


_APP_AGENT_KEY = _fake_jwt("test-app-agent")
_SUBJECT_TYPE = "Person"


def _memory_cache():
    return make_region().configure("dogpile.cache.memory")


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


def _mock_response(mocker, results=_POLICIES):
    """Build a fake httpx2 response returning {"results": results} from .json(), never raising
    on .raise_for_status(). status_code/text are real values since _load() logs both verbatim.
    """
    body = {"results": results}
    response = mocker.Mock(json=mocker.Mock(return_value=body))
    response.status_code = 200
    response.text = json.dumps(body)
    response.raise_for_status = mocker.Mock()
    return response


def _expire(cache):
    """Same deterministic-expiry trick as test_schema.py's _expire."""
    for key, cached in cache.backend._cache.items():
        cache.backend._cache[key] = cached._replace(metadata={**cached.metadata, "ct": 0})


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
    assert client.get_cached(_SUBJECT_TYPE) is None


def test_get_cached_returns_policies_on_success(mocker):
    """A successful fetch returns the results list, unwrapped from the {"results": [...]} envelope."""
    _mock_sync_client(mocker, response=_mock_response(mocker))
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())
    assert client.get_cached(_SUBJECT_TYPE) == _POLICIES


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
    """Same diagnostic logging convention as SchemaClient: request URL and a truncated response
    body preview at DEBUG, so an unexpected/empty policy set can be read straight from the logs.
    """
    caplog.set_level("DEBUG")
    large_policies = [{"policy": {"description": "x" * 600}, "tags": []}]
    _mock_sync_client(mocker, response=_mock_response(mocker, results=large_policies))
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=_memory_cache())

    assert client.get_cached(_SUBJECT_TYPE) == large_policies

    assert "Requesting policies from https://ikg.example.com/access/v1/policies" in caplog.text
    full_body = json.dumps({"results": large_policies})
    assert f"Policy response: status=200 body[:500]={full_body[:500]!r}" in caplog.text
    assert full_body not in caplog.text


def test_get_cached_retries_then_succeeds(mocker):
    """A transient failure on an earlier attempt must not fail the whole fetch - only exhausting
    every retry does.
    """
    error = httpx2.ConnectTimeout("connect timed out")
    client_mock = _mock_sync_client(mocker, side_effect=[error, error, _mock_response(mocker)])
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3, cache=_memory_cache())

    assert client.get_cached(_SUBJECT_TYPE) == _POLICIES
    assert client_mock.get.call_count == _RETRY_ATTEMPTS


def test_get_cached_returns_none_after_exhausting_retries(mocker, caplog):
    """Every attempt failing must return None (extraction proceeds without policy context, see
    subject_binding.py) rather than raise - and must be logged at WARNING once retries are
    exhausted, not ERROR: a policies outage is never fatal here.
    """
    caplog.set_level("WARNING")
    error = httpx2.ConnectTimeout("connect timed out")
    client_mock = _mock_sync_client(mocker, side_effect=error)
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=3, cache=_memory_cache())

    assert client.get_cached(_SUBJECT_TYPE) is None
    assert client_mock.get.call_count == _RETRY_ATTEMPTS
    assert any(record.levelname == "WARNING" for record in caplog.records)
    assert "after 3 attempts" in caplog.text


def test_get_cached_caches_per_subject_type(mocker):
    """Two different subject_types must each trigger their own fetch and be cached separately -
    a shared cache entry would leak one subject_type's policies into another's request.
    """
    other_policies = [{"policy": {"subject": {"type": "Company"}, "actions": ["CAN_SELL"]}, "tags": []}]
    client_mock = _mock_sync_client(
        mocker, side_effect=[_mock_response(mocker), _mock_response(mocker, results=other_policies)]
    )
    cache = _memory_cache()
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)

    assert client.get_cached("Person") == _POLICIES
    assert client.get_cached("Company") == other_policies
    assert client_mock.get.call_count == _FETCH_COUNT_PER_SUBJECT_TYPE

    # Both now cache hits - no further fetches for either subject_type.
    assert client.get_cached("Person") == _POLICIES
    assert client.get_cached("Company") == other_policies
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
    sharing a cache region must not read/write each other's policies.
    """
    shared_cache = _memory_cache()
    other_policies = [{"policy": {"actions": ["CAN_SELL"]}, "tags": []}]

    _mock_sync_client(mocker, response=_mock_response(mocker))
    client_a = PolicyClient("https://a.example.com", _fake_jwt("gid-a"), cache=shared_cache)
    assert client_a.get_cached(_SUBJECT_TYPE) == _POLICIES

    _mock_sync_client(mocker, response=_mock_response(mocker, results=other_policies))
    client_b = PolicyClient("https://b.example.com", _fake_jwt("gid-b"), cache=shared_cache)
    assert client_b.get_cached(_SUBJECT_TYPE) == other_policies

    assert client_a.get_cached(_SUBJECT_TYPE) == _POLICIES


def test_get_cached_refreshes_on_demand_when_entry_expired(mocker):
    """An expired cache entry must trigger a fresh fetch and re-cache, not just sit stale."""
    cache = make_region().configure("dogpile.cache.memory", expiration_time=100)
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, cache=cache)
    assert client.get_cached(_SUBJECT_TYPE) == _POLICIES
    assert client_mock.get.call_count == 1

    _expire(cache)

    assert client.get_cached(_SUBJECT_TYPE) == _POLICIES
    assert client_mock.get.call_count == _EXPECTED_FETCH_COUNT_AFTER_REFRESH


def test_get_cached_returns_none_when_refresh_fails_after_expiry(mocker):
    """Unlike the schema (which falls back to a stale-but-present value), an expired policy
    entry whose refresh fails returns None - there is no preload-time "last known good" to
    preserve for a given subject_type, and a failed refresh is exactly as costly to retry on
    the next request as serving something stale would have been.
    """
    cache = make_region().configure("dogpile.cache.memory", expiration_time=100)
    client_mock = _mock_sync_client(mocker, response=_mock_response(mocker))
    client = PolicyClient("https://ikg.example.com", _APP_AGENT_KEY, retry_attempts=1, cache=cache)
    assert client.get_cached(_SUBJECT_TYPE) == _POLICIES
    _expire(cache)

    client_mock.get.side_effect = httpx2.HTTPError("platform unreachable")

    assert client.get_cached(_SUBJECT_TYPE) is None
