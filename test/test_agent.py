import asyncio
import json
from pathlib import Path

import httpx2
import jwt
import pytest
from dogpile.cache.backends.memory import MemoryBackend
from dogpile.cache.backends.redis import RedisBackend

from indykite.intent_agent.agent import IntentAgent, _build_schema_cache, build_app
from indykite.intent_agent.config import Settings
from indykite.intent_agent.errors import (
    SubjectTypeMismatchError,
    schema_not_populated_error,
    schema_unavailable_error,
    subject_type_mismatch_error,
)
from indykite.intent_agent.input_schema import build_input_schema
from indykite.intent_agent.skills import discover_skills
from indykite.intent_agent.subject_binding import (
    _bind_subject,
    _evaluation_unresolved,
    normalize_intent_response,
    schema_has_no_node_types,
    validate_subject_type,
)

_MIN_EXPECTED_ROUTES = 2
_EXPECTED_BARE_ARRAY_EVALUATION_COUNT = 2
_REDIS_PORT = 6379
_SCHEMA_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_ikg_schema.json"
_EMPTY_SCHEMA_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_ikg_schema_empty.json"

# A real (unsigned, alg=none) JWT - Settings requires IK_APP_AGENT_KEY to be set, and
# SchemaClient requires any non-blank value to be a well-formed JWT (see test_schema.py).
_APP_AGENT_KEY = jwt.encode({"sub": "test-app-agent"}, key="", algorithm="none")


def _load_schema_fixture():
    """Load the real captured IKG schema example used for schema-shape tests."""
    return json.loads(_SCHEMA_FIXTURE_PATH.read_text())


def _load_empty_schema_fixture():
    """Load a real response captured from GET /data-schema/v1 against a genuinely empty
    IndyKite environment (RC, no data model configured yet) - graph.nodes contains only the
    platform's own _Application bootstrap node, not an actually-empty {} (see
    schema_has_no_node_types).
    """
    return json.loads(_EMPTY_SCHEMA_FIXTURE_PATH.read_text())


def test_discovers_extract_intent_skill():
    """The extract-intent SKILL.md must be discovered from the skills/ directory."""
    skills = discover_skills()
    names = {s["name"] for s in skills}
    assert "extract-intent" in names


def test_build_app_registers_routes(monkeypatch, mocker):
    """build_app must wire up the agent-card and JSON-RPC routes on the app."""
    settings = _settings_with_subject_type(monkeypatch)
    fake_schema_client = mocker.Mock(
        preload=mocker.Mock(),
        get_cached=mocker.Mock(return_value=_load_schema_fixture()),
    )

    app, agent = build_app(settings, schema_client=fake_schema_client)

    assert agent.settings.gemini_api_key.get_secret_value() == "test-key"
    assert agent.get_cached_schema() is not None
    assert len(app.routes) >= _MIN_EXPECTED_ROUTES


def test_build_app_starts_when_schema_unavailable_at_cold_start(monkeypatch, mocker):
    """A schema fetch failure at cold start must not abort startup - the platform may still be
    coming up, so the process keeps serving and retries the fetch on every request instead.
    """
    settings = _settings_with_subject_type(monkeypatch)
    # INDYKITE_BASE_URL is now mandatory, so simulate "schema unavailable" via a failed fetch
    # instead of an unset base URL - same client-patching approach as test_schema.py.
    failing_client = mocker.MagicMock()
    failing_client.__enter__ = mocker.Mock(return_value=failing_client)
    failing_client.__exit__ = mocker.Mock(return_value=False)
    failing_client.get = mocker.Mock(side_effect=httpx2.ConnectTimeout("connect timed out"))
    mocker.patch("indykite.intent_agent.schema.httpx2.Client", return_value=failing_client)

    app, agent = build_app(settings)

    assert agent.get_cached_schema() is None
    assert len(app.routes) >= _MIN_EXPECTED_ROUTES


def test_bind_subject():
    data = {"evaluations": [{"action": {"name": "buy"}}]}
    _bind_subject(data, subject_id="user-1", subject_type="Customer")
    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}


def test_bind_subject_overwrites_llm_supplied_subject():
    """An LLM-supplied subject.id/type must be overwritten, not trusted - closes the injection surface."""
    data = {"evaluations": [{"subject": {"type": "Admin", "id": "attacker-controlled"}}]}
    _bind_subject(data, subject_id="user-1", subject_type="Customer")
    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}


def test_bind_subject_applies_to_every_evaluation():
    """Every evaluation in the list must get the override, not just the first."""
    data = {"evaluations": [{"action": {"name": "buy"}}, {"action": {"name": "sell"}}]}
    _bind_subject(data, subject_id="user-1", subject_type="Customer")
    assert all(e["subject"] == {"type": "Customer", "id": "user-1"} for e in data["evaluations"])


def test_bind_subject_tolerates_non_dict_evaluation_entries():
    """A malformed (non-dict) evaluation entry must be skipped, not crash the whole response."""
    data = {"evaluations": ["not-a-dict", {"action": {"name": "buy"}}]}
    _bind_subject(data, subject_id="user-1", subject_type="Customer")
    assert data["evaluations"][0] == "not-a-dict"
    assert data["evaluations"][1]["subject"] == {"type": "Customer", "id": "user-1"}


def test_bind_subject_omits_id_when_subject_id_is_falsy():
    """A falsy subject_id (e.g. Subject Search's blank id) must remove any id, not set it to None/empty."""
    data = {"evaluations": [{"subject": {"type": "Admin", "id": "stale-id"}}]}
    _bind_subject(data, subject_id=None, subject_type="Customer")
    assert data["evaluations"][0]["subject"] == {"type": "Customer"}


@pytest.mark.parametrize(
    ("evaluation", "expected"),
    [
        pytest.param({"action": {"name": "buy"}}, True, id="missing_resource_type"),
        pytest.param(
            {"action": {"name": "buy"}, "resource": {"type": "Stock"}},
            False,
            id="resource_search_missing_id",
        ),
        pytest.param(
            {"action": {"name": ""}, "resource": {"type": "Stock"}},
            True,
            id="empty_action_name_when_action_key_present",
        ),
        pytest.param({"resource": {"type": "Stock", "id": "AAPL"}}, False, id="missing_action_key"),
        pytest.param(
            {"subject": {"type": ""}, "action": {"name": "read"}, "resource": {"type": "Document"}},
            True,
            id="empty_subject_type_when_subject_key_present",
        ),
        pytest.param(
            {"action": {"name": "buy"}, "resource": {"type": "Stock", "id": "AAPL"}},
            False,
            id="missing_subject_key",
        ),
    ],
)
def test_evaluation_unresolved(evaluation, expected):
    """resource.type is core to every evaluation kind (PRD) - never legitimately omitted, so
    its absence always flags. action/subject are only checked when their key is present at
    all: an entirely missing key is a legitimate per-evaluation-kind omission (Action Search
    omits `action`; everything but Subject Search omits `subject` - SKILL.md), while a key
    present with an empty type/name means the LLM tried and failed to correlate it.
    """
    assert _evaluation_unresolved(evaluation) is expected


def test_normalize_intent_response_errors_when_resource_type_unresolved():
    """A resource.type the LLM couldn't correlate to the schema (left absent) must produce
    the same no_intent_error as an empty evaluations array, not a bare success response with
    a hollow resource.
    """
    llm_text = '{"prompt": "buy some birds", "evaluations": [{"action": {"name": "buy"}}]}'
    result = normalize_intent_response(llm_text, "buy some birds", subject_id="user-1", subject_type="Customer")
    data = json.loads(result)
    assert "evaluations" not in data
    assert data["error"] == "Unable to extract intent from prompt."


def test_normalize_intent_response_binds_subject_end_to_end():
    """normalize_intent_response must apply the subject override as part of normalizing LLM output."""
    llm_text = (
        '{"prompt": "buy AAPL", "evaluations": '
        '[{"action": {"name": "buy"}, "resource": {"type": "Stock", "id": "AAPL"}}]}'
    )
    result = normalize_intent_response(llm_text, "buy AAPL", subject_id="user-1", subject_type="Customer")
    data = json.loads(result)
    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}


def test_normalize_intent_response_strips_json_code_fence():
    """A ```json-fenced LLM response must be unwrapped before parsing."""
    llm_text = (
        '```json\n{"prompt": "buy AAPL", "evaluations": '
        '[{"action": {"name": "buy"}, "resource": {"type": "Stock", "id": "AAPL"}}]}\n```'
    )
    result = normalize_intent_response(llm_text, "buy AAPL", subject_id="user-1", subject_type="Customer")
    data = json.loads(result)
    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}


def test_normalize_intent_response_handles_bare_evaluations_array():
    """A bare JSON array (LLM forgot the {"prompt", "evaluations"} wrapper) must be used as the
    evaluations list directly, not nested as a single non-dict evaluation that gets dropped.
    """
    llm_text = (
        '[{"action": {"name": "buy"}, "resource": {"type": "Stock", "id": "AAPL"}}, '
        '{"action": {"name": "sell"}, "resource": {"type": "Stock", "id": "AAPL"}}]'
    )
    result = normalize_intent_response(llm_text, "buy or sell AAPL", subject_id="user-1", subject_type="Customer")
    data = json.loads(result)
    assert len(data["evaluations"]) == _EXPECTED_BARE_ARRAY_EVALUATION_COUNT
    assert all(e["subject"] == {"type": "Customer", "id": "user-1"} for e in data["evaluations"])


def test_normalize_intent_response_errors_on_empty_evaluations():
    """An empty evaluations list must never reach the caller as a bare success response."""
    llm_text = '{"prompt": "What is the distance to the Sun?", "evaluations": []}'
    result = normalize_intent_response(
        llm_text, "What is the distance to the Sun?", subject_id="user-1", subject_type="Customer"
    )
    data = json.loads(result)
    assert "evaluations" not in data
    assert data["error"] == "Unable to extract intent from prompt."
    assert data["reason"]
    assert data["obligation"]


def test_normalize_intent_response_errors_on_malformed_json(caplog):
    """Non-JSON LLM output must fail gracefully with the error shape, not leak raw output to the caller."""
    caplog.set_level("ERROR")
    llm_text = "Sorry, I can't help with that."
    result = normalize_intent_response(llm_text, "some prompt", subject_id="user-1", subject_type="Customer")
    data = json.loads(result)
    assert "evaluations" not in data
    assert "raw_response" not in data
    assert data["error"] == "Unable to extract intent from prompt."
    assert any(record.levelname == "ERROR" for record in caplog.records)


def _settings_with_subject_type(monkeypatch, subject_type="Customer"):
    """Build a Settings with every required env var set, for run_extraction-level tests."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("SUBJECT_TYPE", subject_type)
    monkeypatch.setenv("IK_APP_AGENT_KEY", _APP_AGENT_KEY)
    monkeypatch.setenv("INDYKITE_BASE_URL", "https://ikg.example.com")
    return Settings(_env_file=None)


def test_build_schema_cache_uses_memory_backend_by_default(monkeypatch):
    """CACHE_BACKEND defaults to memory - the in-process Memory cache - with no REDIS_URL set."""
    settings = _settings_with_subject_type(monkeypatch)
    assert isinstance(_build_schema_cache(settings).backend, MemoryBackend)


def test_build_schema_cache_uses_redis_backend_when_configured(monkeypatch):
    """CACHE_BACKEND=redis must select the Redis-backed cache, passing REDIS_URL through as-is
    (not decomposed into host/port) so username/password/db/TLS scheme survive, with
    expiration_time set from SCHEMA_REFRESH_TTL_MINUTES so get_cached() refreshes on-demand.
    """
    redis_url = f"rediss://user:pw@redis.example.com:{_REDIS_PORT}/2"
    monkeypatch.setenv("CACHE_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("SCHEMA_REFRESH_TTL_MINUTES", "2.0")
    settings = _settings_with_subject_type(monkeypatch)
    cache = _build_schema_cache(settings)
    assert isinstance(cache.backend, RedisBackend)
    assert cache.backend.url == redis_url
    assert cache.expiration_time == 2.0 * 60


def test_get_data_schema_tool_returns_cached_schema(monkeypatch, mocker):
    """The get_data_schema tool must return the cached schema as minified JSON."""
    settings = _settings_with_subject_type(monkeypatch)
    schema = _load_schema_fixture()
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock()))
    fake_schema_client = mocker.Mock(get_cached=mocker.Mock(return_value=schema))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client)

    tool = next(t for t in agent.tools if t.name == "get_data_schema")
    result = tool.invoke({})

    assert json.loads(result) == schema


def test_run_extraction_binds_structured_output_schema_derived_from_current_schema(monkeypatch, mocker):
    """run_extraction must call bind_tools with a response_schema derived from the last
    successfully derived schema (via build_input_schema), not some other/unrelated schema.
    """
    settings = _settings_with_subject_type(monkeypatch)
    schema = _load_schema_fixture()
    fake_response = mocker.Mock(text="ok", content="ok", tool_calls=[])
    mock_ainvoke = mocker.AsyncMock(return_value=fake_response)
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    mock_bind_tools = mocker.Mock(return_value=fake_bound_llm)
    fake_llm = mocker.Mock(bind_tools=mock_bind_tools)
    fake_schema_client = mocker.Mock(get_cached=mocker.Mock(return_value=schema))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client)

    asyncio.run(agent.run_extraction("buy AAPL", "user-1"))

    mock_bind_tools.assert_called_once_with(
        agent.tools,
        response_mime_type="application/json",
        response_schema=build_input_schema(schema),
    )


def test_run_extraction_rebuilds_schema_when_cached_schema_changes_between_requests(monkeypatch, mocker):
    """TDD 4.6: the structured-output schema must be derived "each time it reloads" - not
    fixed once (e.g. at __init__ or on the first request). Proven here by having get_cached()
    return a DIFFERENT schema on the second call and checking bind_tools' second call reflects
    it; a schema fixed at construction time would bind the first schema both times.
    """
    settings = _settings_with_subject_type(monkeypatch)
    schema_v1 = _load_schema_fixture()
    schema_v2 = json.loads(json.dumps(schema_v1))  # deep copy, safe to mutate independently
    schema_v2["graph"]["nodes"]["Boat"] = schema_v2["graph"]["nodes"]["Car"]

    fake_response = mocker.Mock(text="ok", content="ok", tool_calls=[])
    mock_ainvoke = mocker.AsyncMock(return_value=fake_response)
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    mock_bind_tools = mocker.Mock(return_value=fake_bound_llm)
    fake_llm = mocker.Mock(bind_tools=mock_bind_tools)
    fake_schema_client = mocker.Mock(get_cached=mocker.Mock(side_effect=[schema_v1, schema_v2]))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client)

    asyncio.run(agent.run_extraction("buy AAPL", "user-1"))
    asyncio.run(agent.run_extraction("rent a boat", "user-1"))

    first_call_schema = mock_bind_tools.call_args_list[0].kwargs["response_schema"]
    second_call_schema = mock_bind_tools.call_args_list[1].kwargs["response_schema"]
    assert first_call_schema == build_input_schema(schema_v1)
    assert second_call_schema == build_input_schema(schema_v2)
    assert first_call_schema != second_call_schema


@pytest.mark.parametrize(
    "case",
    [
        pytest.param((lambda: None, "Customer", schema_unavailable_error, None), id="schema_unavailable"),
        pytest.param(
            (_load_empty_schema_fixture, "Customer", schema_not_populated_error, None),
            id="schema_not_populated",
        ),
        pytest.param(
            (_load_schema_fixture, "NotARealType", subject_type_mismatch_error, "NotARealType"),
            id="subject_type_mismatch",
        ),
    ],
)
def test_run_extraction_fails_before_any_llm_call(monkeypatch, mocker, caplog, case):
    """Schema unavailability, an unpopulated data model, and a real SUBJECT_TYPE mismatch are
    three distinct 4.7 error shapes, each checked and returned before any LLM call - not
    conflated with each other (see schema_has_no_node_types), and not wasting an LLM call on a
    result that would just be discarded. A mismatch's specific detail (log_detail) must reach
    the server-side log but never the caller-facing response - the other two have no such
    detail to leak in the first place.
    """
    build_schema, subject_type, expected_error, log_detail = case
    caplog.set_level("ERROR")
    settings = _settings_with_subject_type(monkeypatch, subject_type=subject_type)
    mock_ainvoke = mocker.AsyncMock()
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=fake_bound_llm))
    fake_schema_client = mocker.Mock(get_cached=mocker.Mock(return_value=build_schema()))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client)

    result = asyncio.run(agent.run_extraction("buy AAPL", "user-1"))

    assert json.loads(result) == expected_error("buy AAPL")
    mock_ainvoke.assert_not_called()
    assert any(record.levelname == "ERROR" for record in caplog.records)
    if log_detail:
        assert log_detail in caplog.text
        assert log_detail not in result


@pytest.mark.parametrize("subject_type", ["Customer", "Company", "_Application", "Car"])
def test_validate_subject_type_accepts_any_real_node_type(subject_type):
    """Every node type in the real schema fixture must be accepted"""
    schema = _load_schema_fixture()
    validate_subject_type(schema, subject_type)


def test_validate_subject_type_rejects_unknown_type():
    """A SUBJECT_TYPE that doesn't match any node type at all must be rejected."""
    schema = _load_schema_fixture()
    with pytest.raises(SubjectTypeMismatchError):
        validate_subject_type(schema, "NotARealType")


def test_validate_subject_type_tolerates_non_dict_nodes():
    """A malformed schema where graph.nodes isn't a dict must be rejected cleanly, not crash."""
    with pytest.raises(SubjectTypeMismatchError):
        validate_subject_type({"graph": {"nodes": ["not", "a", "dict"]}}, "Customer")


def test_validate_subject_type_skips_when_schema_unavailable():
    """With no schema (unset/unreachable INDYKITE_BASE_URL), there's nothing to validate against."""
    validate_subject_type(None, "AnythingGoes")


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        pytest.param({"graph": {"nodes": {}, "edges": []}}, True, id="literally_empty"),
        pytest.param(_load_empty_schema_fixture(), True, id="real_empty_graph_response_bootstrap_node_only"),
        pytest.param(
            {"graph": {"nodes": {"_Application": {}, "Customer": {}}, "edges": []}},
            False,
            id="real_type_alongside_bootstrap_node",
        ),
        pytest.param(_load_schema_fixture(), False, id="real_populated_schema"),
    ],
)
def test_schema_has_no_node_types(schema, expected):
    """Only _Application - the calling app's own node, always present even before any real
    data model is configured - doesn't count as populated; any other node type does, even if
    SUBJECT_TYPE itself doesn't match it (that's a SubjectTypeMismatchError, not this).
    """
    assert schema_has_no_node_types(schema) is expected


def test_build_app_starts_when_schema_not_yet_populated_at_cold_start(monkeypatch, mocker, caplog):
    """A schema that fetches successfully but is empty at cold start must not log a
    SubjectTypeMismatchError - that would misleadingly suggest SUBJECT_TYPE is misconfigured
    when the real cause is just an unpopulated data model.
    """
    caplog.set_level("WARNING")
    settings = _settings_with_subject_type(monkeypatch)
    empty_schema = _load_empty_schema_fixture()
    fake_schema_client = mocker.Mock(
        preload=mocker.Mock(),
        get_cached=mocker.Mock(return_value=empty_schema),
    )

    app, agent = build_app(settings, schema_client=fake_schema_client)

    assert agent.get_cached_schema() == empty_schema
    assert not any(record.levelname == "ERROR" for record in caplog.records)
    assert any("isn't populated yet" in record.message for record in caplog.records)
    assert len(app.routes) >= _MIN_EXPECTED_ROUTES


def test_build_app_starts_when_subject_type_mismatches_schema(monkeypatch, mocker, caplog):
    """build_app must not raise/abort at startup when SUBJECT_TYPE doesn't match the schema -
    the mismatch is instead surfaced per-request (see test_run_extraction_fails_before_any_llm_call
    below), so a bad SUBJECT_TYPE doesn't take down an otherwise-healthy process. It must still
    be logged at startup though - an early signal beats waiting for the first live request.
    """
    caplog.set_level("ERROR")
    settings = _settings_with_subject_type(monkeypatch, subject_type="NotARealType")
    fake_schema_client = mocker.Mock(
        preload=mocker.Mock(),
        get_cached=mocker.Mock(return_value=_load_schema_fixture()),
    )

    app, agent = build_app(settings, schema_client=fake_schema_client)

    assert agent.get_cached_schema() is not None
    assert any(record.levelname == "ERROR" for record in caplog.records)
    assert "NotARealType" in caplog.text
    assert len(app.routes) >= _MIN_EXPECTED_ROUTES
