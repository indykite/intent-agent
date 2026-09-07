import asyncio
import json
from pathlib import Path

import httpx2
import jwt
import pytest
from dogpile.cache.backends.memory import MemoryBackend
from dogpile.cache.backends.redis import RedisBackend
from langchain_core.messages import HumanMessage, ToolMessage
from starlette.testclient import TestClient

from indykite.intent_agent.agent import IntentAgent, _build_schema_cache, build_app
from indykite.intent_agent.audit.message import Outcome, schema_version
from indykite.intent_agent.config import Settings
from indykite.intent_agent.errors import (
    IntentValidationError,
    llm_timeout_error,
    no_intent_error,
    schema_not_populated_error,
    schema_unavailable_error,
    validation_feedback_error,
)
from indykite.intent_agent.input_schema import build_input_schema
from indykite.intent_agent.skills import discover_skills
from indykite.intent_agent.subject_binding import (
    _bind_subject,
    normalize_intent_response,
    schema_has_no_node_types,
    validate_evaluations,
)

_MIN_EXPECTED_ROUTES = 2
_EXPECTED_BARE_ARRAY_EVALUATION_COUNT = 2
_REDIS_PORT = 6379
# Every LLM call run_extraction makes for one prompt: one that activates extract-intent, one
# final answer per attempt.
_LLM_CALLS_TO_FIRST_ANSWER = 2
_LLM_CALLS_WITH_ONE_RETRY = 3
# A tool-call budget no test will exhaust, so a request-deadline test can't hit it first.
_UNREACHABLE_TOOL_BUDGET = 100
_ACTIVATE_EXTRACT_INTENT_CALL = {"name": "activate_skill", "args": {"name": "extract-intent"}, "id": "call-1"}
# Vocabulary from test/fixtures/mock_ikg_schema.json: nodes Car/Company/Customer/_Application,
# edge relations OWN/RENT/WORKS_AT. subject.type is on-schema in both so the off-schema
# answer's failure stays isolated to the action/resource values it's actually testing.
_ON_SCHEMA_ANSWER = (
    '{"prompt": "rent kitt", "evaluations": [{"subject": {"type": "Customer"}, '
    '"action": {"name": "RENT"}, "resource": {"type": "Car", "id": "kitt"}}]}'
)
_OFF_SCHEMA_ANSWER = (
    '{"prompt": "buy AAPL", "evaluations": [{"subject": {"type": "Customer"}, '
    '"action": {"name": "BUY"}, "resource": {"type": "Stock", "id": "AAPL"}}]}'
)
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
    settings = _full_settings(monkeypatch)
    fake_schema_client = mocker.Mock(
        preload=mocker.Mock(),
        get_cached=mocker.Mock(return_value=_load_schema_fixture()),
    )

    app, agent = build_app(settings, schema_client=fake_schema_client, audit_sender=_fake_audit_sender(mocker))

    assert agent.settings.gemini_api_key.get_secret_value() == "test-key"
    assert agent.get_cached_schema() is not None
    assert len(app.routes) >= _MIN_EXPECTED_ROUTES


def test_build_app_closes_audit_sender_on_shutdown(monkeypatch, mocker):
    """The app's lifespan closes the audit sender on shutdown - that is what drains a buffered
    (AUDIT_ASYNC) queue before the process exits, so nothing audited is lost on a rollout.
    """
    settings = _settings_with_subject_type(monkeypatch)
    fake_schema_client = mocker.Mock(preload=mocker.Mock(), get_cached=mocker.Mock(return_value=_load_schema_fixture()))
    sender = _fake_audit_sender(mocker)
    app, _agent = build_app(settings, schema_client=fake_schema_client, audit_sender=sender)

    with TestClient(app):
        sender.close.assert_not_awaited()
    sender.close.assert_awaited_once()


def test_build_app_starts_when_schema_unavailable_at_cold_start(monkeypatch, mocker):
    """A schema fetch failure at cold start must not abort startup - the platform may still be
    coming up, so the process keeps serving and retries the fetch on every request instead.
    """
    settings = _full_settings(monkeypatch)
    # INDYKITE_BASE_URL is now mandatory, so simulate "schema unavailable" via a failed fetch
    # instead of an unset base URL - same client-patching approach as test_schema.py.
    failing_client = mocker.MagicMock()
    failing_client.__enter__ = mocker.Mock(return_value=failing_client)
    failing_client.__exit__ = mocker.Mock(return_value=False)
    failing_client.get = mocker.Mock(side_effect=httpx2.ConnectTimeout("connect timed out"))
    mocker.patch("indykite.intent_agent.schema.httpx2.Client", return_value=failing_client)

    app, agent = build_app(settings, audit_sender=_fake_audit_sender(mocker))

    assert agent.get_cached_schema() is None
    assert len(app.routes) >= _MIN_EXPECTED_ROUTES


def test_bind_subject():
    data = {"evaluations": [{"subject": {"type": "Customer"}, "action": {"name": "buy"}}]}
    _bind_subject(data, subject_id="user-1")
    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}


def test_bind_subject_overwrites_llm_supplied_id_but_not_type():
    """An LLM-supplied subject.id must be overwritten (closes the injection surface where the
    LLM could otherwise set its own identity) - subject.type is left alone, since the LLM
    decides it now, not deployment config.
    """
    data = {"evaluations": [{"subject": {"type": "Customer", "id": "attacker-controlled"}}]}
    _bind_subject(data, subject_id="user-1")
    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}


def test_bind_subject_applies_to_every_evaluation():
    """Every evaluation in the list must get the override, not just the first."""
    data = {
        "evaluations": [
            {"subject": {"type": "Customer"}, "action": {"name": "buy"}},
            {"subject": {"type": "Customer"}, "action": {"name": "sell"}},
        ],
    }
    _bind_subject(data, subject_id="user-1")
    assert all(e["subject"] == {"type": "Customer", "id": "user-1"} for e in data["evaluations"])


def test_bind_subject_tolerates_non_dict_evaluation_entries():
    """A malformed (non-dict) evaluation entry must be skipped, not crash the whole response."""
    data = {"evaluations": ["not-a-dict", {"subject": {"type": "Customer"}, "action": {"name": "buy"}}]}
    _bind_subject(data, subject_id="user-1")
    assert data["evaluations"][0] == "not-a-dict"
    assert data["evaluations"][1]["subject"] == {"type": "Customer", "id": "user-1"}


def test_bind_subject_omits_id_when_subject_id_is_falsy():
    """A falsy subject_id (e.g. Subject Search's blank id) must remove any id, not set it to
    None/empty - subject.type (the search target, for Subject Search) is left untouched.
    """
    data = {"evaluations": [{"subject": {"type": "Admin", "id": "stale-id"}}]}
    _bind_subject(data, subject_id=None)
    assert data["evaluations"][0]["subject"] == {"type": "Admin"}


def test_bind_subject_skips_evaluation_with_no_subject_object():
    """An evaluation without a subject dict is left alone, not given one - unreachable in
    practice since validate_evaluations rejects a missing subject before this ever runs.
    """
    data = {"evaluations": [{"action": {"name": "buy"}}]}
    _bind_subject(data, subject_id="user-1")
    assert "subject" not in data["evaluations"][0]


@pytest.mark.parametrize(
    ("evaluation", "expected_errors"),
    [
        pytest.param(
            {"subject": {"type": "Customer"}, "action": {"name": "RENT"}},
            ["resource is missing or not an object"],
            id="missing_resource",
        ),
        pytest.param(
            {"subject": {"type": "Customer"}, "action": {"name": "RENT"}, "resource": {"type": "Car"}},
            [],
            id="resource_search_missing_id",
        ),
        pytest.param(
            {"subject": {"type": "Customer"}, "action": {"name": ""}, "resource": {"type": "Car"}},
            ["action.name is missing or blank"],
            id="empty_action_name_when_action_key_present",
        ),
        pytest.param(
            {"subject": {"type": "Customer"}, "resource": {"type": "Car", "id": "kitt"}},
            [],
            id="missing_action_key",
        ),
        pytest.param(
            {"subject": {"type": ""}, "action": {"name": "OWN"}, "resource": {"type": "Car"}},
            ["subject.type is missing or blank"],
            id="empty_subject_type_when_subject_key_present",
        ),
        pytest.param(
            {"action": {"name": "RENT"}, "resource": {"type": "Car", "id": "kitt"}},
            ["subject is missing or not an object"],
            id="missing_subject_key",
        ),
        pytest.param(
            {"subject": "Admin", "action": {"name": "OWN"}, "resource": {"type": "Car"}},
            ["subject is missing or not an object"],
            id="subject_not_an_object",
        ),
        pytest.param(
            {"subject": {"type": "Customer"}, "action": {"name": "BUY"}, "resource": {"type": "Car"}},
            ["action.name 'BUY' is not defined in the data schema"],
            id="off_schema_action_name",
        ),
        pytest.param(
            {"subject": {"type": "Customer"}, "action": {"name": "RENT"}, "resource": {"type": "Stock"}},
            ["resource.type 'Stock' is not defined in the data schema"],
            id="off_schema_resource_type",
        ),
        pytest.param(
            {"subject": {"type": "Admin"}, "action": {"name": "OWN"}, "resource": {"type": "Car"}},
            ["subject.type 'Admin' is not defined in the data schema"],
            id="off_schema_subject_type",
        ),
        pytest.param(
            {"subject": {"type": "Customer"}, "action": "RENT", "resource": {"type": "Car"}},
            ["action is not an object"],
            id="action_not_an_object",
        ),
        pytest.param("not-a-dict", ["evaluation is not an object"], id="evaluation_not_an_object"),
        pytest.param(
            {"subject": {"type": "Admin"}, "action": {"name": "BUY"}, "resource": {"type": "Stock"}},
            [
                "resource.type 'Stock' is not defined in the data schema",
                "subject.type 'Admin' is not defined in the data schema",
                "action.name 'BUY' is not defined in the data schema",
            ],
            id="every_field_off_schema_reports_all",
        ),
    ],
)
def test_validate_evaluations(evaluation, expected_errors):
    """resource.type and subject.type are core to every evaluation kind (PRD; subject.type is
    the LLM's own call now, not deployment config) - never legitimately omitted, so their
    absence always flags. action is the one field only checked when its key is present at
    all: an entirely missing key is a legitimate per-evaluation-kind omission (Action Search
    omits `action` - SKILL.md), while a key present with an empty or off-schema type/name
    means the LLM tried and failed to correlate it (TDD 4.8 schema-membership check). Every
    problem is reported, not just the first, so one retry can fix them all.
    """
    errors = validate_evaluations(_load_schema_fixture(), [evaluation])
    assert errors == [f"evaluations[0]: {error}" for error in expected_errors]


def test_validate_evaluations_indexes_each_evaluation():
    """Errors must name the offending evaluation's index so the LLM can tell which entry to fix."""
    evaluations = [
        {"subject": {"type": "Customer"}, "action": {"name": "RENT"}, "resource": {"type": "Car"}},
        {"subject": {"type": "Customer"}, "action": {"name": "BUY"}, "resource": {"type": "Car"}},
    ]
    assert validate_evaluations(_load_schema_fixture(), evaluations) == [
        "evaluations[1]: action.name 'BUY' is not defined in the data schema",
    ]


def test_validate_evaluations_skips_action_membership_when_schema_has_no_edges():
    """A schema with nodes but no edges yet leaves action.name unconstrained in the structured
    output (input_schema._constrain_to_enum no-op) - the post-assembly check must mirror that,
    or every request with an action would fail against such a schema.
    """
    schema = _load_schema_fixture()
    schema["graph"]["edges"] = []
    evaluations = [{"subject": {"type": "Customer"}, "action": {"name": "ANYTHING"}, "resource": {"type": "Car"}}]
    assert validate_evaluations(schema, evaluations) == []


def test_validate_evaluations_tolerates_malformed_schema_sections():
    """Non-dict nodes / non-list edges must read as an empty vocabulary (no membership check),
    not crash - same tolerance as input_schema.build_input_schema.
    """
    schema = {"graph": {"nodes": "oops", "edges": {"also": "oops"}}}
    evaluations = [{"subject": {"type": "Customer"}, "action": {"name": "RENT"}, "resource": {"type": "Car"}}]
    assert validate_evaluations(schema, evaluations) == []


def test_validate_evaluations_ignores_non_string_relations():
    """A malformed relation (a number, an empty string) is not part of the action vocabulary -
    the same rule input_schema applies when building the enum, so the two never disagree.
    """
    schema = _load_schema_fixture()
    schema["graph"]["edges"] = [{"relation": 123}, {"relation": ""}, {"relation": "RENT"}]
    evaluations = [
        {"subject": {"type": "Customer"}, "action": {"name": "RENT"}, "resource": {"type": "Car"}},
        {"subject": {"type": "Customer"}, "action": {"name": "123"}, "resource": {"type": "Car"}},
    ]
    assert validate_evaluations(schema, evaluations) == [
        "evaluations[1]: action.name '123' is not defined in the data schema",
    ]


def test_validation_feedback_error_lists_every_error_in_fixed_shape():
    """The follow-up message must be the fixed 4.7 error shape and carry each error verbatim
    in `reason`, one per line, plus the corrective instruction in `obligation` - not just the
    first error, or the LLM can only fix one thing per retry.
    """
    feedback = validation_feedback_error("rent kitt", ["evaluations[0]: first", "evaluations[1]: second"])
    assert set(feedback) == {"prompt", "error", "reason", "obligation"}
    assert feedback["prompt"] == "rent kitt"
    assert feedback["error"] == no_intent_error("rent kitt")["error"]
    assert "- evaluations[0]: first\n- evaluations[1]: second" in feedback["reason"]
    assert "empty evaluations array" in feedback["obligation"]


def _normalize(llm_text, prompt="rent kitt"):
    """normalize_intent_response against the fixture schema with the usual test subject id."""
    return normalize_intent_response(
        llm_text,
        prompt,
        schema=_load_schema_fixture(),
        subject_id="user-1",
    )


def test_normalize_intent_response_raises_when_resource_type_unresolved():
    """A resource.type the LLM couldn't correlate to the schema (left absent) is a validation
    failure to retry - not a bare success response with a hollow resource, and not the
    non-retryable no_intent_error reserved for an empty evaluations array.
    """
    llm_text = '{"prompt": "buy some birds", "evaluations": [{"action": {"name": "RENT"}}]}'
    with pytest.raises(IntentValidationError, match="resource is missing"):
        _normalize(llm_text, "buy some birds")


def test_normalize_intent_response_raises_when_values_off_schema():
    """Off-schema action.name / resource.type must raise (TDD 4.8) with every problem listed,
    so run_extraction can feed them all back to the LLM in one retry.
    """
    with pytest.raises(IntentValidationError) as excinfo:
        _normalize(_OFF_SCHEMA_ANSWER, "buy AAPL")
    assert excinfo.value.errors == [
        "evaluations[0]: resource.type 'Stock' is not defined in the data schema",
        "evaluations[0]: action.name 'BUY' is not defined in the data schema",
    ]


@pytest.mark.parametrize(
    "evaluations_json",
    [
        pytest.param('{"action": {"name": "RENT"}}', id="non_empty_object"),
        pytest.param("{}", id="empty_object"),
        pytest.param("0", id="zero"),
        pytest.param('""', id="empty_string"),
        pytest.param("null", id="null"),
        pytest.param(None, id="missing_key"),
    ],
)
def test_normalize_intent_response_raises_when_evaluations_not_an_array(evaluations_json):
    """Only an actual empty array is the LLM's "nothing coherent" signal. Any other value -
    including falsy ones like {}, 0, "", null, or the key missing entirely - is malformed
    output that must be retried (IntentValidationError), never silently accepted as a
    non-retryable no_intent_error.
    """
    llm_text = (
        '{"prompt": "rent kitt"}'
        if evaluations_json is None
        else f'{{"prompt": "rent kitt", "evaluations": {evaluations_json}}}'
    )
    with pytest.raises(IntentValidationError, match="evaluations is missing or not an array"):
        _normalize(llm_text)


@pytest.mark.parametrize("llm_text", ['"just a string"', "42", "true", "null"])
def test_normalize_intent_response_raises_on_bare_json_scalar(llm_text):
    """Valid JSON that is neither an object nor an array (the one tolerated wrapper mistake)
    is malformed output to retry, not an empty evaluations list to report as no-intent.
    """
    with pytest.raises(IntentValidationError, match="not a JSON object"):
        _normalize(llm_text)


def test_normalize_intent_response_binds_subject_end_to_end():
    """normalize_intent_response must apply the subject override as part of normalizing LLM output."""
    data = json.loads(_normalize(_ON_SCHEMA_ANSWER))
    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}


def test_normalize_intent_response_strips_json_code_fence():
    """A ```json-fenced LLM response must be unwrapped before parsing."""
    data = json.loads(_normalize(f"```json\n{_ON_SCHEMA_ANSWER}\n```"))
    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}


def test_normalize_intent_response_handles_bare_evaluations_array():
    """A bare JSON array (LLM forgot the {"prompt", "evaluations"} wrapper) must be used as the
    evaluations list directly, not nested as a single non-dict evaluation that gets dropped.
    """
    llm_text = (
        '[{"subject": {"type": "Customer"}, "action": {"name": "RENT"}, "resource": {"type": "Car", "id": "kitt"}}, '
        '{"subject": {"type": "Customer"}, "action": {"name": "OWN"}, "resource": {"type": "Car", "id": "kitt"}}]'
    )
    data = json.loads(_normalize(llm_text, "rent or own kitt"))
    assert data["prompt"] == "rent or own kitt"
    assert len(data["evaluations"]) == _EXPECTED_BARE_ARRAY_EVALUATION_COUNT
    assert all(e["subject"] == {"type": "Customer", "id": "user-1"} for e in data["evaluations"])


def test_normalize_intent_response_errors_on_empty_evaluations():
    """An empty evaluations list must never reach the caller as a bare success response - but
    it's the LLM's legitimate "nothing coherent" signal (SKILL.md), so it's the error shape
    directly, not a validation failure to retry.
    """
    llm_text = '{"prompt": "What is the distance to the Sun?", "evaluations": []}'
    data = json.loads(_normalize(llm_text, "What is the distance to the Sun?"))
    assert "evaluations" not in data
    assert data["error"] == "Unable to extract intent from prompt."
    assert data["reason"]
    assert data["obligation"]


def test_normalize_intent_response_raises_on_malformed_json():
    """Non-JSON LLM output is a validation failure to retry (TDD 8: "malformed or off-schema"
    share one row) - never leaked to the caller as raw output.
    """
    with pytest.raises(IntentValidationError, match="not valid JSON"):
        _normalize("Sorry, I can't help with that.", "some prompt")


def _full_settings(monkeypatch):
    """Build a Settings with every required env var set, for run_extraction-level tests.
    AUDIT_DELIVERY is required too; the file path is never created here because these tests
    build an IntentAgent directly (no sender) or inject a fake sender into build_app.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("IK_APP_AGENT_KEY", _APP_AGENT_KEY)
    monkeypatch.setenv("INDYKITE_BASE_URL", "https://ikg.example.com")
    monkeypatch.setenv("AUDIT_DELIVERY", "file")
    monkeypatch.setenv("AUDIT_FILE_STORAGE_PATH", "./.private/audit-never-created")
    monkeypatch.setenv("AUDIT_FILE_FORMAT", "json")
    monkeypatch.setenv("AUDIT_FILE_ROTATION_STRATEGY", "time")
    return Settings(_env_file=None)


def _fake_audit_sender(mocker):
    """An audit sender that records nothing - build_app tests here are about startup, not auditing."""
    return mocker.Mock(send=mocker.AsyncMock(), close=mocker.AsyncMock())


def test_build_schema_cache_uses_memory_backend_by_default(monkeypatch):
    """CACHE_BACKEND defaults to memory - the in-process Memory cache - with no REDIS_URL set."""
    settings = _full_settings(monkeypatch)
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
    settings = _full_settings(monkeypatch)
    cache = _build_schema_cache(settings)
    assert isinstance(cache.backend, RedisBackend)
    assert cache.backend.url == redis_url
    assert cache.expiration_time == 2.0 * 60


def test_get_data_schema_tool_returns_cached_schema(monkeypatch, mocker):
    """The get_data_schema tool must return the cached schema as minified JSON."""
    settings = _full_settings(monkeypatch)
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
    settings = _full_settings(monkeypatch)
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
    settings = _full_settings(monkeypatch)
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
        pytest.param((lambda: None, schema_unavailable_error), id="schema_unavailable"),
        pytest.param((_load_empty_schema_fixture, schema_not_populated_error), id="schema_not_populated"),
    ],
)
def test_run_extraction_fails_before_any_llm_call(monkeypatch, mocker, caplog, case):
    """Schema unavailability and an unpopulated data model are two distinct 4.7 error shapes,
    each checked and returned before any LLM call - not conflated with each other (see
    schema_has_no_node_types), and not wasting an LLM call on a result that would just be
    discarded.
    """
    build_schema, expected_error = case
    caplog.set_level("ERROR")
    settings = _full_settings(monkeypatch)
    mock_ainvoke = mocker.AsyncMock()
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=fake_bound_llm))
    fake_schema_client = mocker.Mock(get_cached=mocker.Mock(return_value=build_schema()))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client)

    result = asyncio.run(agent.run_extraction("buy AAPL", "user-1"))

    assert json.loads(result.text) == expected_error("buy AAPL")
    assert result.outcome == Outcome.ERROR
    assert result.message == expected_error("buy AAPL")["reason"]
    mock_ainvoke.assert_not_called()
    assert any(record.levelname == "ERROR" for record in caplog.records)
    if log_detail:
        assert log_detail in caplog.text
        assert log_detail not in result.text


def _llm_response(mocker, text="", tool_calls=()):
    """A fake LLM response: plain text, optionally with tool calls to dispatch first."""
    return mocker.Mock(text=text, content=text, tool_calls=list(tool_calls))


def _agent_with_responses(monkeypatch, mocker, responses, **env):
    """An IntentAgent over the fixture schema whose LLM answers with `responses` in order.

    `responses` entries are either fake response objects or awaitables (side effects);
    `env` sets extra env vars (e.g. VALIDATION_MAX_RETRIES) before Settings is built.
    """
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    settings = _full_settings(monkeypatch)
    mock_ainvoke = mocker.AsyncMock(side_effect=responses)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock(ainvoke=mock_ainvoke)))
    fake_schema_client = mocker.Mock(get_cached=mocker.Mock(return_value=_load_schema_fixture()))
    return IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client), mock_ainvoke


def test_run_extraction_returns_bound_evaluations_when_first_answer_is_on_schema(monkeypatch, mocker):
    """The happy path: skill activation, one on-schema answer, subject bound - no retry spent."""
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [_llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]), _llm_response(mocker, _ON_SCHEMA_ANSWER)],
    )

    result = asyncio.run(agent.run_extraction("rent kitt", "user-1"))
    data = json.loads(result.text)

    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}
    assert mock_ainvoke.call_count == _LLM_CALLS_TO_FIRST_ANSWER
    # The audit classification (TDD 4.9) rides along with the response text.
    assert result.outcome == Outcome.SUCCESS
    assert result.evaluations == data["evaluations"]
    assert result.validation_retries == 0
    assert result.schema_version == schema_version(_load_schema_fixture())
    assert result.message == ""


def test_run_extraction_retries_with_feedback_when_answer_is_off_schema(monkeypatch, mocker, caplog):
    """TDD 4.9: an off-schema answer is fed back to the LLM as a follow-up message naming the
    bad values, in the same conversation (no re-activation of the skill), and the corrected
    answer is what the caller gets.
    """
    caplog.set_level("WARNING")
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [
            _llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]),
            _llm_response(mocker, _OFF_SCHEMA_ANSWER),
            _llm_response(mocker, _ON_SCHEMA_ANSWER),
        ],
    )

    result = asyncio.run(agent.run_extraction("buy AAPL", "user-1"))
    data = json.loads(result.text)

    assert data["evaluations"][0] == {
        "action": {"name": "RENT"},
        "resource": {"type": "Car", "id": "kitt"},
        "subject": {"type": "Customer", "id": "user-1"},
    }
    assert result.outcome == Outcome.SUCCESS
    assert result.validation_retries == 1
    assert mock_ainvoke.call_count == _LLM_CALLS_WITH_ONE_RETRY
    messages = mock_ainvoke.call_args_list[-1].args[0]
    feedback = messages[-1]
    assert isinstance(feedback, HumanMessage)
    assert json.loads(feedback.content) == validation_feedback_error(
        "buy AAPL",
        [
            "evaluations[0]: resource.type 'Stock' is not defined in the data schema",
            "evaluations[0]: action.name 'BUY' is not defined in the data schema",
        ],
    )
    assert "attempt 1/2 failed validation" in caplog.text


def test_run_extraction_fails_after_retry_budget_when_answers_stay_off_schema(monkeypatch, mocker, caplog):
    """Past VALIDATION_MAX_RETRIES the caller gets no_intent_error - never the off-schema
    output, and never a third LLM call with the default N=1.
    """
    caplog.set_level("ERROR")
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [
            _llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]),
            _llm_response(mocker, _OFF_SCHEMA_ANSWER),
            _llm_response(mocker, _OFF_SCHEMA_ANSWER),
        ],
    )

    result = asyncio.run(agent.run_extraction("buy AAPL", "user-1"))

    assert json.loads(result.text) == no_intent_error("buy AAPL")
    assert "Stock" not in result.text
    assert mock_ainvoke.call_count == _LLM_CALLS_WITH_ONE_RETRY
    assert "after 1 retries" in caplog.text
    # TDD 4.9: terminal failure after N retries is audited as validation_failed, N recorded.
    assert result.outcome == Outcome.VALIDATION_FAILED
    assert result.validation_retries == 1
    assert result.message == no_intent_error("buy AAPL")["reason"]


def test_run_extraction_retry_budget_is_configurable(monkeypatch, mocker):
    """VALIDATION_MAX_RETRIES=0 must disable retry entirely: one answer, then the error."""
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [_llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]), _llm_response(mocker, _OFF_SCHEMA_ANSWER)],
        VALIDATION_MAX_RETRIES=0,
    )

    result = asyncio.run(agent.run_extraction("buy AAPL", "user-1"))

    assert json.loads(result.text) == no_intent_error("buy AAPL")
    assert result.outcome == Outcome.VALIDATION_FAILED
    assert result.validation_retries == 0
    assert mock_ainvoke.call_count == _LLM_CALLS_TO_FIRST_ANSWER


def test_run_extraction_retries_malformed_json_too(monkeypatch, mocker):
    """Non-JSON output shares the retry path with off-schema output (TDD 8) - one corrective
    round trip, then the corrected answer.
    """
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [
            _llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]),
            _llm_response(mocker, "Sorry, I can't help with that."),
            _llm_response(mocker, _ON_SCHEMA_ANSWER),
        ],
    )

    data = json.loads(asyncio.run(agent.run_extraction("rent kitt", "user-1")).text)

    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}
    assert mock_ainvoke.call_count == _LLM_CALLS_WITH_ONE_RETRY


def test_run_extraction_passes_free_text_through_when_skill_not_activated(monkeypatch, mocker):
    """Without extract-intent activated, the answer isn't an extraction - it's passed through
    unvalidated, and an empty answer becomes the placeholder text. Neither extracted an
    intent, so both audit as `unmatched` with the pass-through text as the message.
    """
    agent, _ = _agent_with_responses(monkeypatch, mocker, [_llm_response(mocker, "hello")])
    result = asyncio.run(agent.run_extraction("hi", "user-1"))
    assert result.text == "hello"
    assert result.outcome == Outcome.UNMATCHED
    assert result.message == "hello"

    agent, _ = _agent_with_responses(monkeypatch, mocker, [_llm_response(mocker, "")])
    result = asyncio.run(agent.run_extraction("hi", "user-1"))
    assert result.text == "(No response generated)"
    assert result.outcome == Outcome.UNMATCHED


def test_run_extraction_gives_up_when_tool_call_budget_runs_out(monkeypatch, mocker, caplog):
    """An LLM that keeps calling tools past TOOL_CALL_MAX_ITERATIONS never yields an answer to
    validate. Once extract-intent is active that's a failed extraction: the fixed error shape
    (no_intent_error), not the free-text placeholder, and not a retry (which would just
    re-spend the budget).
    """
    caplog.set_level("ERROR")
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [_llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL])] * 2,
        TOOL_CALL_MAX_ITERATIONS=1,
    )

    result = asyncio.run(agent.run_extraction("rent kitt", "user-1"))

    assert json.loads(result.text) == no_intent_error("rent kitt")
    assert result.outcome == Outcome.ERROR
    assert mock_ainvoke.call_count == 1
    assert "Tool-call budget (1) exhausted" in caplog.text


def test_run_extraction_keeps_placeholder_when_budget_runs_out_before_activation(monkeypatch, mocker):
    """Without extract-intent ever activated there's no extraction to report on, so budget
    exhaustion stays the plain placeholder - the same pass-through as any non-extraction answer.
    """
    get_schema_call = {"name": "get_data_schema", "args": {}, "id": "call-2"}
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [_llm_response(mocker, tool_calls=[get_schema_call])] * 2,
        TOOL_CALL_MAX_ITERATIONS=1,
    )

    assert asyncio.run(agent.run_extraction("hi", "user-1")).text == "(No response generated)"
    assert mock_ainvoke.call_count == 1


def test_run_extraction_short_circuits_repeat_skill_activation(monkeypatch, mocker, caplog):
    """A second activate_skill for the already-active extract-intent must get a short "answer
    now" tool message, not the full skill body plus schema again - re-injecting those every
    round is what an LLM stuck re-activating until the budget runs out looks like live.
    """
    caplog.set_level("INFO")
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [
            _llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]),
            _llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]),
            _llm_response(mocker, _ON_SCHEMA_ANSWER),
        ],
    )

    data = json.loads(asyncio.run(agent.run_extraction("rent kitt", "user-1")).text)

    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}
    assert mock_ainvoke.call_count == _LLM_CALLS_WITH_ONE_RETRY
    messages = mock_ainvoke.call_args_list[-1].args[0]
    tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]
    # One real activation (skill body + schema), then one short-circuit (reminder only).
    assert ["<data_schema>" in result for result in tool_results] == [True, False]
    assert ["already active" in result for result in tool_results] == [False, True]
    assert caplog.text.count("Activating skill: extract-intent") == 1


def test_run_extraction_returns_timeout_error_when_llm_call_exceeds_limit(monkeypatch, mocker, caplog):
    """A single LLM call past LLM_TIMEOUT_SECONDS must surface as llm_timeout_error - not hang
    the request, and not be retried.
    """
    caplog.set_level("ERROR")

    async def _hang(_messages):
        await asyncio.sleep(10)

    agent, mock_ainvoke = _agent_with_responses(monkeypatch, mocker, _hang, LLM_TIMEOUT_SECONDS=0.01)

    result = asyncio.run(agent.run_extraction("rent kitt", "user-1"))

    assert json.loads(result.text) == llm_timeout_error("rent kitt")
    assert result.outcome == Outcome.ERROR
    assert result.message == llm_timeout_error("rent kitt")["reason"]
    assert mock_ainvoke.call_count == 1
    assert "exceeded 0.0s" in caplog.text


def test_run_extraction_enforces_request_deadline_across_calls(monkeypatch, mocker, caplog):
    """REQUEST_TIMEOUT_SECONDS caps the whole extraction: calls that each stay under
    LLM_TIMEOUT_SECONDS but keep coming (tool-call rounds, retries) are cut off together at
    the request deadline, surfacing as llm_timeout_error with the end-to-end log line.
    """
    caplog.set_level("ERROR")

    async def _slow_tool_call(_messages):
        await asyncio.sleep(0.02)
        return _llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL])

    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        _slow_tool_call,
        LLM_TIMEOUT_SECONDS=0.1,
        REQUEST_TIMEOUT_SECONDS=0.1,
        TOOL_CALL_MAX_ITERATIONS=_UNREACHABLE_TOOL_BUDGET,
    )

    result = asyncio.run(agent.run_extraction("rent kitt", "user-1"))

    assert json.loads(result.text) == llm_timeout_error("rent kitt")
    # Several rounds got through before the deadline, but nowhere near the tool budget.
    assert 1 < mock_ainvoke.call_count < _UNREACHABLE_TOOL_BUDGET
    assert "Request exceeded the 0.1s end-to-end budget" in caplog.text
    assert "LLM call exceeded" not in caplog.text


def test_run_extraction_times_out_during_retry_too(monkeypatch, mocker):
    """LLM_TIMEOUT_SECONDS bounds every call, the validation retry included: a retry that
    hangs surfaces as llm_timeout_error - never the rejected off-schema answer, and never a
    request that outlives the PRD's end-to-end budget.
    """
    scripted = [
        _llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]),
        _llm_response(mocker, _OFF_SCHEMA_ANSWER),
    ]

    async def _answer_then_hang(_messages):
        if scripted:
            return scripted.pop(0)
        await asyncio.sleep(10)
        return None

    agent, mock_ainvoke = _agent_with_responses(monkeypatch, mocker, _answer_then_hang, LLM_TIMEOUT_SECONDS=0.01)

    result = asyncio.run(agent.run_extraction("buy AAPL", "user-1"))

    assert json.loads(result.text) == llm_timeout_error("buy AAPL")
    assert "Stock" not in result.text
    assert mock_ainvoke.call_count == _LLM_CALLS_WITH_ONE_RETRY


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
    data model is configured - doesn't count as populated; any other node type does.
    """
    assert schema_has_no_node_types(schema) is expected


def test_build_app_starts_when_schema_not_yet_populated_at_cold_start(monkeypatch, mocker, caplog):
    """A schema that fetches successfully but is empty at cold start must not abort startup -
    the data model may just not be populated yet.
    """
    caplog.set_level("WARNING")
    settings = _full_settings(monkeypatch)
    empty_schema = _load_empty_schema_fixture()
    fake_schema_client = mocker.Mock(
        preload=mocker.Mock(),
        get_cached=mocker.Mock(return_value=empty_schema),
    )

    app, agent = build_app(settings, schema_client=fake_schema_client, audit_sender=_fake_audit_sender(mocker))

    assert agent.get_cached_schema() == empty_schema
    assert not any(record.levelname == "ERROR" for record in caplog.records)
    assert any("isn't populated yet" in record.message for record in caplog.records)
    assert len(app.routes) >= _MIN_EXPECTED_ROUTES
