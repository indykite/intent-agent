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
from indykite.intent_agent.errors import SubjectTypeMismatchError
from indykite.intent_agent.skills import discover_skills
from indykite.intent_agent.subject_binding import _bind_subject, normalize_intent_response, validate_subject_type

_MIN_EXPECTED_ROUTES = 2
_EXPECTED_BARE_ARRAY_EVALUATION_COUNT = 2
_REDIS_PORT = 6379
_SCHEMA_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_ikg_schema.json"

# A real (unsigned, alg=none) JWT - Settings requires IK_APP_AGENT_KEY to be set, and
# SchemaClient requires any non-blank value to be a well-formed JWT (see test_schema.py).
_APP_AGENT_KEY = jwt.encode({"sub": "test-app-agent"}, key="", algorithm="none")


def _load_schema_fixture():
    """Load the real captured IKG schema example used for schema-shape tests."""
    return json.loads(_SCHEMA_FIXTURE_PATH.read_text())


def test_discovers_extract_intent_skill():
    """The extract-intent SKILL.md must be discovered from the skills/ directory."""
    skills = discover_skills()
    names = {s["name"] for s in skills}
    assert "extract-intent" in names


def test_build_app_registers_routes(monkeypatch, mocker):
    """build_app must wire up the agent-card and JSON-RPC routes on the app."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("SUBJECT_TYPE", "Customer")
    monkeypatch.setenv("IK_APP_AGENT_KEY", _APP_AGENT_KEY)
    monkeypatch.setenv("INDYKITE_BASE_URL", "https://ikg.example.com")
    settings = Settings(_env_file=None)
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
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("SUBJECT_TYPE", "Customer")
    monkeypatch.setenv("INDYKITE_BASE_URL", "https://ikg.example.com")
    monkeypatch.setenv("IK_APP_AGENT_KEY", _APP_AGENT_KEY)
    settings = Settings(_env_file=None)
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


def test_normalize_intent_response_binds_subject_end_to_end():
    """normalize_intent_response must apply the subject override as part of normalizing LLM output."""
    llm_text = '{"prompt": "buy AAPL", "evaluations": [{"action": {"name": "buy"}}]}'
    result = normalize_intent_response(llm_text, "buy AAPL", subject_id="user-1", subject_type="Customer")
    data = json.loads(result)
    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}


def test_normalize_intent_response_strips_json_code_fence():
    """A ```json-fenced LLM response must be unwrapped before parsing."""
    llm_text = '```json\n{"prompt": "buy AAPL", "evaluations": [{"action": {"name": "buy"}}]}\n```'
    result = normalize_intent_response(llm_text, "buy AAPL", subject_id="user-1", subject_type="Customer")
    data = json.loads(result)
    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}


def test_normalize_intent_response_handles_bare_evaluations_array():
    """A bare JSON array (LLM forgot the {"prompt", "evaluations"} wrapper) must be used as the
    evaluations list directly, not nested as a single non-dict evaluation that gets dropped.
    """
    llm_text = '[{"action": {"name": "buy"}}, {"action": {"name": "sell"}}]'
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


def test_normalize_intent_response_passes_through_llm_supplied_error():
    """If the LLM already followed SKILL.md and emitted its own error/reason/obligation, keep it as-is."""
    llm_text = json.dumps(
        {
            "prompt": "Can Joe do stuff in the folder?",
            "error": "Unable to extract intent from prompt.",
            "reason": "The prompt is unclear.",
            "obligation": "Ask the prompter to clarify their request.",
        }
    )
    result = normalize_intent_response(
        llm_text, "Can Joe do stuff in the folder?", subject_id="user-1", subject_type="Customer"
    )
    data = json.loads(result)
    assert data["reason"] == "The prompt is unclear."
    assert "evaluations" not in data


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


def test_run_extraction_skips_llm_when_schema_unavailable(monkeypatch, mocker, caplog):
    """Schema unavailability must be checked before any LLM call, not after wasting one."""
    caplog.set_level("ERROR")
    settings = _settings_with_subject_type(monkeypatch)
    mock_ainvoke = mocker.AsyncMock()
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=fake_bound_llm))
    fake_schema_client = mocker.Mock(get_cached=mocker.Mock(return_value=None))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client)

    result = asyncio.run(agent.run_extraction("buy AAPL", "user-1"))

    data = json.loads(result)
    assert "error" in data
    assert "evaluations" not in data
    mock_ainvoke.assert_not_called()
    assert any(record.levelname == "ERROR" for record in caplog.records)


def test_run_extraction_fails_with_message_when_subject_type_mismatches_schema(monkeypatch, mocker, caplog):
    """A SUBJECT_TYPE that isn't a DigitalTwin type in the schema must fail the request with the
    4.7 error shape - not crash the process and not reach the LLM. The response's "reason" must
    stay generic (not leak the schema's actual DigitalTwin type names to the API caller) - the
    specific detail belongs in the server-side log only.
    """
    caplog.set_level("ERROR")
    settings = _settings_with_subject_type(monkeypatch, subject_type="NotARealType")
    mock_ainvoke = mocker.AsyncMock()
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=fake_bound_llm))
    fake_schema_client = mocker.Mock(get_cached=mocker.Mock(return_value=_load_schema_fixture()))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client)

    result = asyncio.run(agent.run_extraction("buy AAPL", "user-1"))

    data = json.loads(result)
    assert "error" in data
    assert "evaluations" not in data
    assert "NotARealType" not in data["reason"]
    mock_ainvoke.assert_not_called()
    assert any(record.levelname == "ERROR" for record in caplog.records)
    assert "NotARealType" in caplog.text


@pytest.mark.parametrize("subject_type", ["Customer", "Company", "_Application"])
def test_validate_subject_type_accepts_real_digital_twin_types(subject_type):
    """Each DigitalTwin-labelled node type in the real schema fixture must be accepted."""
    schema = _load_schema_fixture()
    validate_subject_type(schema, subject_type)


def test_validate_subject_type_rejects_non_digital_twin_type():
    """Car is a real node type in the fixture, but not DigitalTwin-labelled - must be rejected."""
    schema = _load_schema_fixture()
    with pytest.raises(SubjectTypeMismatchError):
        validate_subject_type(schema, "Car")


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


def test_build_app_starts_when_subject_type_mismatches_schema(monkeypatch, mocker, caplog):
    """build_app must not raise/abort at startup when SUBJECT_TYPE doesn't match the schema -
    the mismatch is instead surfaced per-request (see test_run_extraction_fails_with_message...
    below), so a bad SUBJECT_TYPE doesn't take down an otherwise-healthy process. It must still
    be logged at startup though - an early signal beats waiting for the first live request.
    """
    caplog.set_level("ERROR")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("SUBJECT_TYPE", "NotARealType")
    monkeypatch.setenv("IK_APP_AGENT_KEY", _APP_AGENT_KEY)
    monkeypatch.setenv("INDYKITE_BASE_URL", "https://ikg.example.com")
    settings = Settings(_env_file=None)
    fake_schema_client = mocker.Mock(
        preload=mocker.Mock(),
        get_cached=mocker.Mock(return_value=_load_schema_fixture()),
    )

    app, agent = build_app(settings, schema_client=fake_schema_client)

    assert agent.get_cached_schema() is not None
    assert any(record.levelname == "ERROR" for record in caplog.records)
    assert "NotARealType" in caplog.text
    assert len(app.routes) >= _MIN_EXPECTED_ROUTES
