import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx2
import jwt
import pytest
from dogpile.cache.backends.memory import MemoryBackend
from dogpile.cache.backends.redis import RedisBackend
from langchain_core.messages import HumanMessage, ToolMessage
from starlette.testclient import TestClient

from indykite.intent_agent.agent import IntentAgent, RequestSubject, _policy_block
from indykite.intent_agent.app import build_app
from indykite.intent_agent.audit.message import Outcome, schema_version
from indykite.intent_agent.config import Settings
from indykite.intent_agent.errors import (
    llm_timeout_error,
    no_intent_error,
    policy_unavailable_error,
    request_timeout_error,
    schema_not_populated_error,
    schema_unavailable_error,
    subject_type_mismatch_error,
    validation_feedback_error,
)
from indykite.intent_agent.input_schema import build_input_schema, render_schema_summary
from indykite.intent_agent.schema import build_schema_cache
from indykite.intent_agent.skills import discover_skills
from test.test_extraction_helpers import (
    _MOCK_POLICIES,
    _OFF_SCHEMA_ANSWER,
    _ON_SCHEMA_ANSWER,
    _RATING,
    _SCHEMA_FIXTURE_PATH,
    _load_empty_schema_fixture,
    _load_schema_fixture,
    _policy,
    _rated,
    _stamped,
)

_MIN_EXPECTED_ROUTES = 2

_REDIS_PORT = 6379

# One LLM call activates extract-intent, then one final answer per attempt.
_LLM_CALLS_TO_FIRST_ANSWER = 2

_LLM_CALLS_WITH_ONE_RETRY = 3

# A tool-call budget no test will exhaust, so a request-deadline test can't hit it first.
_UNREACHABLE_TOOL_BUDGET = 100

# Well under the 0.3s blocked sleep below, with headroom for scheduler jitter.
_LOOP_NOT_BLOCKED_THRESHOLD_SECONDS = 0.15

_ACTIVATE_EXTRACT_INTENT_CALL = {"name": "activate_skill", "args": {"name": "extract-intent"}, "id": "call-1"}

# SchemaClient requires IK_APP_AGENT_KEY to be a well-formed JWT; unsigned is enough.
_APP_AGENT_KEY = jwt.encode({"sub": "test-app-agent"}, key="", algorithm="none")


def _fake_schema_client(mocker, **kwargs):
    """A fake SchemaClient; run_extraction only calls get_cached() on it."""
    return mocker.Mock(**kwargs)


def _fake_policy_client(mocker, **kwargs):
    """A fake PolicyClient; run_extraction only calls get_cached() on it."""
    return mocker.Mock(**kwargs)


def test_discovers_extract_intent_skill():
    skills = discover_skills()
    names = {s["name"] for s in skills}
    assert "extract-intent" in names


def test_build_app_registers_routes(monkeypatch, mocker):
    """build_app wires up the agent-card and JSON-RPC routes."""
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
    """Shutdown closes the audit sender (draining an async queue) and this agent's own fetch executor."""
    settings = _full_settings(monkeypatch)
    fake_schema_client = mocker.Mock(preload=mocker.Mock(), get_cached=mocker.Mock(return_value=_load_schema_fixture()))
    sender = _fake_audit_sender(mocker)
    app, agent = build_app(settings, schema_client=fake_schema_client, audit_sender=sender)

    with TestClient(app):
        sender.close.assert_not_awaited()
        assert not agent._fetch_executor._shutdown
    sender.close.assert_awaited_once()
    assert agent._fetch_executor._shutdown


def test_build_app_starts_when_schema_unavailable_at_cold_start(monkeypatch, mocker):
    """A schema fetch failure at cold start doesn't abort startup."""
    settings = _full_settings(monkeypatch)
    # Simulate "schema unavailable" with a failed fetch.
    failing_client = mocker.MagicMock()
    failing_client.__enter__ = mocker.Mock(return_value=failing_client)
    failing_client.__exit__ = mocker.Mock(return_value=False)
    failing_client.get = mocker.Mock(side_effect=httpx2.ConnectTimeout("connect timed out"))
    mocker.patch("indykite.intent_agent.schema.httpx2.Client", return_value=failing_client)

    app, agent = build_app(settings, audit_sender=_fake_audit_sender(mocker))

    assert agent.get_cached_schema() is None
    assert len(app.routes) >= _MIN_EXPECTED_ROUTES


def test_validation_feedback_error_lists_every_error_in_fixed_shape():
    """The feedback uses the fixed error shape and lists every error, one per line, in `reason`."""
    feedback = validation_feedback_error("rent kitt", ["evaluations[0]: first", "evaluations[1]: second"])
    assert set(feedback) == {"prompt", "error", "reason", "obligation"}
    assert feedback["prompt"] == "rent kitt"
    assert feedback["error"] == no_intent_error("rent kitt")["error"]
    assert "- evaluations[0]: first\n- evaluations[1]: second" in feedback["reason"]
    assert "empty evaluations array" in feedback["obligation"]


def test_validation_feedback_error_separates_schema_and_shape_problems():
    """Schema and shape errors go under separate headings; an empty heading is left out."""
    schema_error = "evaluations[0]: action.name 'BUY' is not defined in the data schema"
    blank_error = "evaluations[1]: resource.type is missing or blank"
    shape_error = "evaluations[0]: context.filters is not a string"
    reason = validation_feedback_error("buy AAPL", [shape_error, schema_error, blank_error])["reason"]
    assert (
        reason.index("Values not matched to the data schema or access policies:")
        < reason.index(schema_error)
        < reason.index(blank_error)
    )
    assert reason.index(blank_error) < reason.index("Malformed shape:") < reason.index(shape_error)

    only_shape = validation_feedback_error("buy AAPL", [shape_error])["reason"]
    assert "Values not matched to the data schema or access policies:" not in only_shape
    assert "Malformed shape:" in only_shape


def test_extract_intent_skill_documents_every_context_sub_shape():
    """The skill body keeps an example of each context sub-shape, including the filters reference answer."""
    skill = next(s for s in discover_skills() if s["name"] == "extract-intent")
    for key in ("parameters", "qualifiers"):
        assert f'"{key}":' in skill["body"], f"no {key} example in SKILL.md"
    assert "Are there any Decisions that incorporated the refund policy document" in skill["body"]
    assert "resource.document_type == 'Policy' && resource.name.contains('refund')" in skill["body"]


def test_run_extraction_logs_why_the_model_refused(monkeypatch, mocker, caplog):
    caplog.set_level("DEBUG")
    refusal = '{"explanation": "No policy covers stock purchases.", "evaluations": []}'
    agent, _ = _agent_with_responses(
        monkeypatch,
        mocker,
        [_llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]), _llm_response(mocker, refusal)],
    )

    result = asyncio.run(agent.run_extraction("buy AAPL", RequestSubject(id="user-1", type="Customer")))

    assert json.loads(result.text) == no_intent_error("buy AAPL")
    assert (result.confidence, result.explanation) == ("", "No policy covers stock purchases.")
    assert "Extraction refused for prompt 'buy AAPL': No policy covers stock purchases." in caplog.text


def _full_settings(monkeypatch):
    """Build a Settings with every required env var set, for run_extraction-level tests."""
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
    settings = _full_settings(monkeypatch)
    assert isinstance(build_schema_cache(settings, ThreadPoolExecutor(max_workers=1)).backend, MemoryBackend)


def test_build_schema_cache_uses_redis_backend_when_configured(monkeypatch):
    """CACHE_BACKEND=redis passes REDIS_URL through as-is and sets expiration from SCHEMA_REFRESH_TTL_MINUTES."""
    redis_url = f"rediss://user:pw@redis.example.com:{_REDIS_PORT}/2"
    monkeypatch.setenv("CACHE_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("SCHEMA_REFRESH_TTL_MINUTES", "2.0")
    settings = _full_settings(monkeypatch)
    cache = build_schema_cache(settings, ThreadPoolExecutor(max_workers=1))
    assert isinstance(cache.backend, RedisBackend)
    assert cache.backend.url == redis_url
    assert cache.expiration_time == 2.0 * 60


def test_get_data_schema_tool_returns_cached_schema(monkeypatch, mocker):
    """get_data_schema returns the request's schema snapshot as minified JSON, not a fresh fetch."""
    settings = _full_settings(monkeypatch)
    schema = _load_schema_fixture()
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock()))
    agent = IntentAgent(settings, llm=fake_llm)

    result = agent._make_get_data_schema_tool(schema).invoke({})

    assert json.loads(result) == schema


@pytest.mark.parametrize(
    ("policies", "policy_fetch_failed", "expected_json"),
    [
        pytest.param(
            [{"policy": {"actions": ["APPROVE_REFUND"], "resource": {"type": "Car"}}, "tags": []}],
            False,
            [{"policy": {"actions": ["APPROVE_REFUND"], "resource": {"type": "Car"}}, "tags": []}],
            id="returns_the_given_policies",
        ),
        pytest.param(None, True, [], id="none_returns_empty_list_not_null"),
    ],
)
def test_get_policies_tool_returns_the_request_snapshot(
    monkeypatch, mocker, policies, policy_fetch_failed, expected_json
):
    """get_policies returns the request's policy snapshot as minified JSON; None becomes []."""
    settings = _full_settings(monkeypatch)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock()))
    agent = IntentAgent(settings, llm=fake_llm)

    result = agent._make_get_policies_tool(policies, policy_fetch_failed=policy_fetch_failed).invoke({})

    assert json.loads(result) == expected_json


def test_get_policies_tool_logs_only_when_the_fetch_had_failed(monkeypatch, mocker, caplog):
    """get_policies logs only when policy_fetch_failed is set, even if stale policies are served."""
    caplog.set_level("WARNING")
    settings = _full_settings(monkeypatch)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock()))
    agent = IntentAgent(settings, llm=fake_llm)

    agent._make_get_policies_tool(None, policy_fetch_failed=True).invoke({})
    assert "get_policies invoked but this request's policy fetch had failed" in caplog.text

    caplog.clear()
    stale_policies = [{"policy": {"actions": ["APPROVE_REFUND"], "resource": {"type": "Car"}}, "tags": []}]
    agent._make_get_policies_tool(stale_policies, policy_fetch_failed=True).invoke({})
    assert "get_policies invoked but this request's policy fetch had failed" in caplog.text

    caplog.clear()
    agent._make_get_policies_tool([], policy_fetch_failed=False).invoke({})
    assert caplog.text == ""


def test_run_extraction_binds_structured_output_schema_derived_from_current_schema(monkeypatch, mocker):
    """bind_tools gets the response schema built from the current schema, plus the expected tools (compared by name)."""
    settings = _full_settings(monkeypatch)
    schema = _load_schema_fixture()
    fake_response = mocker.Mock(text="ok", content="ok", tool_calls=[])
    mock_ainvoke = mocker.AsyncMock(return_value=fake_response)
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    mock_bind_tools = mocker.Mock(return_value=fake_bound_llm)
    fake_llm = mocker.Mock(bind_tools=mock_bind_tools)
    fake_schema_client = _fake_schema_client(mocker, get_cached=mocker.Mock(return_value=schema))
    fake_policy_client = _fake_policy_client(mocker, get_cached=mocker.Mock(return_value=(_MOCK_POLICIES, False)))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client, policy_client=fake_policy_client)

    asyncio.run(agent.run_extraction("buy AAPL", RequestSubject(id="user-1", type="Customer")))

    mock_bind_tools.assert_called_once()
    call = mock_bind_tools.call_args
    assert {t.name for t in call.args[0]} == {"activate_skill", "get_data_schema", "get_policies"}
    assert call.kwargs == {
        "response_mime_type": "application/json",
        "response_schema": build_input_schema(schema, _MOCK_POLICIES),
    }


def test_run_extraction_rebuilds_schema_when_cached_schema_changes_between_requests(monkeypatch, mocker):
    """The response schema is rebuilt from each request's schema, not fixed at startup."""
    settings = _full_settings(monkeypatch)
    schema_v1 = _load_schema_fixture()
    schema_v2 = json.loads(json.dumps(schema_v1))  # deep copy, safe to mutate independently
    schema_v2["graph"]["nodes"]["Boat"] = schema_v2["graph"]["nodes"]["Car"]
    policies = [*_MOCK_POLICIES, _policy("Customer", "RENT", "Boat")]

    fake_response = mocker.Mock(text="ok", content="ok", tool_calls=[])
    mock_ainvoke = mocker.AsyncMock(return_value=fake_response)
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    mock_bind_tools = mocker.Mock(return_value=fake_bound_llm)
    fake_llm = mocker.Mock(bind_tools=mock_bind_tools)
    fake_schema_client = _fake_schema_client(mocker, get_cached=mocker.Mock(side_effect=[schema_v1, schema_v2]))
    fake_policy_client = _fake_policy_client(mocker, get_cached=mocker.Mock(return_value=(policies, False)))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client, policy_client=fake_policy_client)

    asyncio.run(agent.run_extraction("buy AAPL", RequestSubject(id="user-1", type="Customer")))
    asyncio.run(agent.run_extraction("rent a boat", RequestSubject(id="user-1", type="Customer")))

    first_call_schema = mock_bind_tools.call_args_list[0].kwargs["response_schema"]
    second_call_schema = mock_bind_tools.call_args_list[1].kwargs["response_schema"]
    assert first_call_schema == build_input_schema(schema_v1, policies)
    assert second_call_schema == build_input_schema(schema_v2, policies)
    assert first_call_schema != second_call_schema


@pytest.mark.parametrize(
    "case",
    [
        pytest.param((lambda: None, "Customer", schema_unavailable_error, None, "ERROR"), id="schema_unavailable"),
        pytest.param(
            (_load_empty_schema_fixture, "Customer", schema_not_populated_error, None, "WARNING"),
            id="schema_not_populated",
        ),
        pytest.param(
            (_load_schema_fixture, "NotARealType", subject_type_mismatch_error, "NotARealType", "ERROR"),
            id="subject_type_mismatch",
        ),
    ],
)
def test_run_extraction_fails_before_any_llm_call(monkeypatch, mocker, caplog, case):
    """Unavailable, unpopulated or mismatched schema each return their own error before any LLM call."""
    build_schema, subject_type, expected_error, log_detail, expected_level = case
    caplog.set_level("WARNING")
    settings = _full_settings(monkeypatch)
    mock_ainvoke = mocker.AsyncMock()
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=fake_bound_llm))
    fake_schema_client = _fake_schema_client(mocker, get_cached=mocker.Mock(return_value=build_schema()))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client)

    result = asyncio.run(agent.run_extraction("buy AAPL", RequestSubject(id="user-1", type=subject_type)))

    assert json.loads(result.text) == expected_error("buy AAPL")
    assert result.outcome == Outcome.ERROR
    assert result.message == expected_error("buy AAPL")["reason"]
    # None, not False: the policy fetch was never attempted.
    assert result.policy_fetch_failed is None
    mock_ainvoke.assert_not_called()
    assert any(record.levelname == expected_level for record in caplog.records)
    if log_detail:
        assert log_detail in caplog.text
        assert log_detail not in result.text


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            (None, True, "Policy fetch failed for subject_type 'Customer' and no policies"), id="fetch_failed"
        ),
        pytest.param(([], False, "No access policies exist for subject_type 'Customer'"), id="none_exist"),
    ],
)
def test_run_extraction_refuses_without_usable_policies(monkeypatch, mocker, caplog, case):
    fetched, fetch_failed, log_detail = case
    caplog.set_level("WARNING")
    settings = _full_settings(monkeypatch)
    mock_ainvoke = mocker.AsyncMock()
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock(ainvoke=mock_ainvoke)))
    fake_schema_client = _fake_schema_client(mocker, get_cached=mocker.Mock(return_value=_load_schema_fixture()))
    fake_policy_client = _fake_policy_client(mocker, get_cached=mocker.Mock(return_value=(fetched, fetch_failed)))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client, policy_client=fake_policy_client)

    result = asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    assert json.loads(result.text) == policy_unavailable_error("rent kitt")
    assert result.outcome == Outcome.ERROR
    assert result.policy_fetch_failed is fetch_failed
    mock_ainvoke.assert_not_called()
    assert any(record.levelname == "WARNING" and log_detail in record.getMessage() for record in caplog.records)


def _llm_response(mocker, text="", tool_calls=()):
    """A fake LLM response; an {evaluations} answer gets _RATING."""
    text = _rated(text)
    return mocker.Mock(text=text, content=text, tool_calls=list(tool_calls))


def _agent_with_responses(monkeypatch, mocker, responses, **env):
    """An IntentAgent over the fixture schema whose LLM answers with `responses` (or awaitables) in order."""
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    settings = _full_settings(monkeypatch)
    mock_ainvoke = mocker.AsyncMock(side_effect=responses)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock(ainvoke=mock_ainvoke)))
    fake_schema_client = _fake_schema_client(mocker, get_cached=mocker.Mock(return_value=_load_schema_fixture()))
    # A fake client, so no real network call; without policies every request would be refused.
    fake_policy_client = _fake_policy_client(mocker, get_cached=mocker.Mock(return_value=(_MOCK_POLICIES, False)))
    return (
        IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client, policy_client=fake_policy_client),
        mock_ainvoke,
    )


def test_run_extraction_returns_bound_evaluations_when_first_answer_is_on_schema(monkeypatch, mocker, caplog):
    """The happy path: one answer, subject bound, rating logged but not returned."""
    caplog.set_level("DEBUG")
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [_llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]), _llm_response(mocker, _ON_SCHEMA_ANSWER)],
    )

    result = asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))
    data = json.loads(result.text)

    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}
    assert mock_ainvoke.call_count == _LLM_CALLS_TO_FIRST_ANSWER
    assert result.outcome == Outcome.SUCCESS
    assert result.evaluations == data["evaluations"]
    assert result.validation_retries == 0
    assert result.schema_version == schema_version(_load_schema_fixture())
    assert result.message == ""
    assert (result.confidence, result.explanation) == (_RATING["confidence"], _RATING["explanation"])
    assert "confidence" not in data
    assert f"Extraction confidence high for prompt 'rent kitt': {_RATING['explanation']}" in caplog.text


def test_run_extraction_retries_with_feedback_when_answer_is_off_schema(monkeypatch, mocker, caplog):
    """An off-schema answer is fed back in the same conversation and the corrected answer is returned."""
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

    result = asyncio.run(agent.run_extraction("buy AAPL", RequestSubject(id="user-1", type="Customer")))
    data = json.loads(result.text)

    assert data["evaluations"][0] == {
        "action": {"name": "RENT"},
        "resource": {"type": "Car", "id": "kitt"},
        "subject": {"type": "Customer", "id": "user-1"},
        "context": _stamped({}),
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
            "evaluations[0]: action.name 'BUY' is not named by any access policy",
        ],
    )
    assert "attempt 1/2 failed validation" in caplog.text


def test_run_extraction_fails_after_retry_budget_when_answers_stay_off_schema(monkeypatch, mocker, caplog):
    """Past VALIDATION_MAX_RETRIES the caller gets no_intent_error, with no extra LLM call."""
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

    result = asyncio.run(agent.run_extraction("buy AAPL", RequestSubject(id="user-1", type="Customer")))

    assert json.loads(result.text) == no_intent_error("buy AAPL")
    assert "Stock" not in result.text
    assert mock_ainvoke.call_count == _LLM_CALLS_WITH_ONE_RETRY
    assert "after 1 retries" in caplog.text
    assert result.outcome == Outcome.VALIDATION_FAILED
    assert result.validation_retries == 1
    assert result.message == no_intent_error("buy AAPL")["reason"]


def test_run_extraction_retry_budget_is_configurable(monkeypatch, mocker):
    """VALIDATION_MAX_RETRIES=0 disables retry."""
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [_llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]), _llm_response(mocker, _OFF_SCHEMA_ANSWER)],
        VALIDATION_MAX_RETRIES=0,
    )

    result = asyncio.run(agent.run_extraction("buy AAPL", RequestSubject(id="user-1", type="Customer")))

    assert json.loads(result.text) == no_intent_error("buy AAPL")
    assert result.outcome == Outcome.VALIDATION_FAILED
    assert result.validation_retries == 0
    assert mock_ainvoke.call_count == _LLM_CALLS_TO_FIRST_ANSWER


def test_run_extraction_retries_malformed_json_too(monkeypatch, mocker):
    """Non-JSON output is retried like off-schema output."""
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [
            _llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]),
            _llm_response(mocker, "Sorry, I can't help with that."),
            _llm_response(mocker, _ON_SCHEMA_ANSWER),
        ],
    )

    data = json.loads(asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer"))).text)

    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}
    assert mock_ainvoke.call_count == _LLM_CALLS_WITH_ONE_RETRY


def test_run_extraction_passes_free_text_through_when_skill_not_activated(monkeypatch, mocker):
    """Without extract-intent activated, the answer passes through unvalidated and audits as unmatched."""
    agent, _ = _agent_with_responses(monkeypatch, mocker, [_llm_response(mocker, "hello")])
    result = asyncio.run(agent.run_extraction("hi", RequestSubject(id="user-1", type="Customer")))
    assert result.text == "hello"
    assert result.outcome == Outcome.UNMATCHED
    assert result.message == "hello"

    agent, _ = _agent_with_responses(monkeypatch, mocker, [_llm_response(mocker, "")])
    result = asyncio.run(agent.run_extraction("hi", RequestSubject(id="user-1", type="Customer")))
    assert result.text == "(No response generated)"
    assert result.outcome == Outcome.UNMATCHED


def test_run_extraction_gives_up_when_tool_call_budget_runs_out(monkeypatch, mocker, caplog):
    """Exhausting the tool budget after activation is no_intent_error, without a retry."""
    caplog.set_level("ERROR")
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [_llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL])] * 2,
        TOOL_CALL_MAX_ITERATIONS=1,
    )

    result = asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    assert json.loads(result.text) == no_intent_error("rent kitt")
    assert result.outcome == Outcome.ERROR
    assert mock_ainvoke.call_count == 1
    assert "Tool-call budget (1) exhausted" in caplog.text


def test_run_extraction_keeps_placeholder_when_budget_runs_out_before_activation(monkeypatch, mocker):
    """Exhausting the tool budget before activation returns the plain placeholder."""
    get_schema_call = {"name": "get_data_schema", "args": {}, "id": "call-2"}
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [_llm_response(mocker, tool_calls=[get_schema_call])] * 2,
        TOOL_CALL_MAX_ITERATIONS=1,
    )

    assert (
        asyncio.run(agent.run_extraction("hi", RequestSubject(id="user-1", type="Customer"))).text
        == "(No response generated)"
    )
    assert mock_ainvoke.call_count == 1


def test_run_extraction_short_circuits_repeat_skill_activation(monkeypatch, mocker, caplog):
    """A repeat activate_skill gets a short "answer now" message, not the skill body and schema again."""
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

    data = json.loads(asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer"))).text)

    assert data["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}
    assert mock_ainvoke.call_count == _LLM_CALLS_WITH_ONE_RETRY
    messages = mock_ainvoke.call_args_list[-1].args[0]
    tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]
    assert ["<data_schema>" in result for result in tool_results] == [True, False]
    assert ["already active" in result for result in tool_results] == [False, True]
    assert caplog.text.count("Activating skill: extract-intent") == 1


@pytest.mark.parametrize(
    ("preview_chars", "shown", "note"),
    [
        pytest.param(100_000, 3, "", id="all_fit"),
        pytest.param(150, 1, "(2 more policies not shown - call get_policies for all of them.)", id="cut_at_a_policy"),
        pytest.param(1, 1, "(2 more policies not shown - call get_policies for all of them.)", id="first_always_whole"),
    ],
)
def test_policy_block_keeps_every_policy_whole(preview_chars, shown, note):
    policies = [_policy("Customer", action, "Car") for action in ("RENT", "OWN", "INSURE")]
    for entry in policies:
        entry["policy"]["condition"] = {"cypher": "MATCH (subject:Customer)-[:OWN]->(resource:Car)"}
    block = _policy_block(policies, preview_chars)
    listed = block.split("<policies>\n", 1)[1].split("\n</policies>", 1)[0].split("\n")
    assert [json.loads(line) for line in listed if line.startswith("{")] == policies[:shown]
    assert (note in block) if note else ("more policies" not in block)


def test_run_extraction_policies_block_injection_on_skill_activation(monkeypatch, mocker):
    cypher = "MATCH (subject:Customer)-[:OWN]->(resource:Car) WHERE resource.model = 'refundable'"
    refund_policy = _policy("Customer", "APPROVE_REFUND", "Car")
    refund_policy["policy"]["condition"] = {"cypher": cypher}
    cached_policies = [*_MOCK_POLICIES, refund_policy]
    settings = _full_settings(monkeypatch)
    mock_ainvoke = mocker.AsyncMock(
        side_effect=[
            _llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]),
            _llm_response(mocker, _ON_SCHEMA_ANSWER),
        ],
    )
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock(ainvoke=mock_ainvoke)))
    fake_schema_client = _fake_schema_client(mocker, get_cached=mocker.Mock(return_value=_load_schema_fixture()))
    fake_policy_client = _fake_policy_client(mocker, get_cached=mocker.Mock(return_value=(cached_policies, False)))
    agent = IntentAgent(
        settings,
        llm=fake_llm,
        schema_client=fake_schema_client,
        policy_client=fake_policy_client,
    )

    asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    messages = mock_ainvoke.call_args_list[-1].args[0]
    tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]
    activation = next(result for result in tool_results if "<policies>" in result)
    assert "- Customer -APPROVE_REFUND-> Car" in activation
    assert json.dumps(cypher)[1:-1] in activation
    assert "condition.cypher" in activation


def test_run_extraction_marks_policy_fetch_failed_when_serving_stale_policies(monkeypatch, mocker):
    """Stale policies from a failed fetch still mark policy_fetch_failed on the audit and the response."""
    settings = _full_settings(monkeypatch)
    stale_policies = _MOCK_POLICIES
    mock_ainvoke = mocker.AsyncMock(
        side_effect=[
            _llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]),
            _llm_response(mocker, _ON_SCHEMA_ANSWER),
        ],
    )
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock(ainvoke=mock_ainvoke)))
    fake_schema_client = _fake_schema_client(mocker, get_cached=mocker.Mock(return_value=_load_schema_fixture()))
    fake_policy_client = _fake_policy_client(mocker, get_cached=mocker.Mock(return_value=(stale_policies, True)))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client, policy_client=fake_policy_client)

    result = asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    assert result.outcome == Outcome.SUCCESS
    assert result.policy_fetch_failed is True
    assert json.loads(result.text)["evaluations"][0]["context"]["parameters"]["policy_fetch_failed"] is True


def test_run_extraction_reuses_one_policy_snapshot_for_the_whole_request(monkeypatch, mocker):
    """Policies are fetched once per request and that snapshot is what the skill injection shows."""
    settings = _full_settings(monkeypatch)
    mock_ainvoke = mocker.AsyncMock(
        side_effect=[
            _llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]),
            _llm_response(mocker, _ON_SCHEMA_ANSWER),
        ],
    )
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock(ainvoke=mock_ainvoke)))
    fake_schema_client = _fake_schema_client(mocker, get_cached=mocker.Mock(return_value=_load_schema_fixture()))
    first_fetch_policies = [*_MOCK_POLICIES, _policy("Customer", "APPROVE_REFUND", "Car")]
    second_fetch_policies = [*_MOCK_POLICIES, _policy("Customer", "ESCALATE", "Car")]
    fake_policy_client = _fake_policy_client(
        mocker, get_cached=mocker.Mock(side_effect=[(first_fetch_policies, False), (second_fetch_policies, False)])
    )
    agent = IntentAgent(
        settings,
        llm=fake_llm,
        schema_client=fake_schema_client,
        policy_client=fake_policy_client,
    )

    asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    assert fake_policy_client.get_cached.call_count == 1
    messages = mock_ainvoke.call_args_list[-1].args[0]
    tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]
    assert any("APPROVE_REFUND" in result for result in tool_results)
    assert not any("ESCALATE" in result for result in tool_results)


def test_run_extraction_reuses_one_schema_snapshot_for_the_whole_request(monkeypatch, mocker):
    """The schema is fetched once per request and that snapshot is what the skill injection shows."""
    settings = _full_settings(monkeypatch)
    mock_ainvoke = mocker.AsyncMock(
        side_effect=[
            _llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]),
            _llm_response(mocker, _ON_SCHEMA_ANSWER),
        ],
    )
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock(ainvoke=mock_ainvoke)))
    first_fetch_schema = _load_schema_fixture()
    second_fetch_schema = json.loads(json.dumps(first_fetch_schema))
    second_fetch_schema["graph"]["nodes"]["TotallyDifferentNodeType"] = {"properties": {}}
    fake_schema_client = _fake_schema_client(
        mocker, get_cached=mocker.Mock(side_effect=[first_fetch_schema, second_fetch_schema])
    )
    fake_policy_client = _fake_policy_client(mocker, get_cached=mocker.Mock(return_value=(_MOCK_POLICIES, False)))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client, policy_client=fake_policy_client)

    asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    assert fake_schema_client.get_cached.call_count == 1
    messages = mock_ainvoke.call_args_list[-1].args[0]
    tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]
    assert not any("TotallyDifferentNodeType" in result for result in tool_results)


def test_run_extraction_returns_timeout_error_when_llm_call_exceeds_limit(monkeypatch, mocker, caplog):
    """An LLM call past LLM_TIMEOUT_SECONDS returns llm_timeout_error without a retry."""
    caplog.set_level("ERROR")

    async def _hang(_messages):
        await asyncio.sleep(10)

    agent, mock_ainvoke = _agent_with_responses(monkeypatch, mocker, _hang, LLM_TIMEOUT_SECONDS=0.01)

    result = asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    assert json.loads(result.text) == llm_timeout_error("rent kitt")
    assert result.outcome == Outcome.ERROR
    assert result.message == llm_timeout_error("rent kitt")["reason"]
    assert mock_ainvoke.call_count == 1
    assert "exceeded 0.0s" in caplog.text


def test_run_extraction_enforces_request_deadline_across_calls(monkeypatch, mocker, caplog):
    """REQUEST_TIMEOUT_SECONDS cuts off a run of individually fast calls with llm_timeout_error."""
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
        SCHEMA_FETCH_TIMEOUT_SECONDS=0.03,
        POLICY_FETCH_TIMEOUT_SECONDS=0.03,
        POLICY_FETCH_RETRY_ATTEMPTS=1,
        TOOL_CALL_MAX_ITERATIONS=_UNREACHABLE_TOOL_BUDGET,
    )

    result = asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    assert json.loads(result.text) == llm_timeout_error("rent kitt")
    # Several rounds got through before the deadline, but nowhere near the tool budget.
    assert 1 < mock_ainvoke.call_count < _UNREACHABLE_TOOL_BUDGET
    assert "Request exceeded the 0.1s end-to-end budget" in caplog.text
    assert "LLM call exceeded" not in caplog.text


def test_run_extraction_timeout_mid_policy_fetch_reports_the_fetch_as_failed(monkeypatch, mocker):
    """A deadline during the policy fetch returns request_timeout_error and audits the fetch as failed."""
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "0.02")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("SCHEMA_FETCH_TIMEOUT_SECONDS", "0.005")
    monkeypatch.setenv("POLICY_FETCH_TIMEOUT_SECONDS", "0.005")
    monkeypatch.setenv("POLICY_FETCH_RETRY_ATTEMPTS", "1")
    settings = _full_settings(monkeypatch)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock(ainvoke=mocker.AsyncMock())))
    schema = _load_schema_fixture()
    fake_schema_client = _fake_schema_client(mocker, get_cached=mocker.Mock(return_value=schema))
    fake_policy_client = _fake_policy_client(
        mocker, get_cached=mocker.Mock(side_effect=lambda _subject_type: time.sleep(0.1) or [])
    )
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client, policy_client=fake_policy_client)

    result = asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    assert result.outcome == Outcome.ERROR
    assert result.policy_fetch_failed is True
    assert result.policy_version == ""
    assert result.schema_version == schema_version(schema)
    assert json.loads(result.text) == request_timeout_error("rent kitt")


def test_run_extraction_timeout_mid_schema_fetch_reports_the_request_timeout(monkeypatch, mocker):
    """A deadline during the schema fetch returns request_timeout_error with schema/policy fields unset."""
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "0.02")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("SCHEMA_FETCH_TIMEOUT_SECONDS", "0.005")
    monkeypatch.setenv("POLICY_FETCH_TIMEOUT_SECONDS", "0.005")
    monkeypatch.setenv("POLICY_FETCH_RETRY_ATTEMPTS", "1")
    settings = _full_settings(monkeypatch)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock(ainvoke=mocker.AsyncMock())))
    fake_schema_client = _fake_schema_client(
        mocker, get_cached=mocker.Mock(side_effect=lambda: time.sleep(0.1) or _load_schema_fixture())
    )
    fake_policy_client = _fake_policy_client(mocker, get_cached=mocker.Mock(return_value=([], False)))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client, policy_client=fake_policy_client)

    result = asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    assert result.outcome == Outcome.ERROR
    assert result.schema_version == ""
    assert result.policy_fetch_failed is None
    assert result.policy_version == ""
    assert json.loads(result.text) == request_timeout_error("rent kitt")


def test_run_extraction_slow_schema_fetch_does_not_block_the_event_loop(monkeypatch, mocker):
    """A slow synchronous schema fetch doesn't block other coroutines on the event loop."""
    settings = _full_settings(monkeypatch)
    mock_ainvoke = mocker.AsyncMock(return_value=_llm_response(mocker, text="no intent here"))
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock(ainvoke=mock_ainvoke)))
    fake_schema_client = _fake_schema_client(
        mocker, get_cached=mocker.Mock(side_effect=lambda: (time.sleep(0.3), _load_schema_fixture())[1])
    )
    fake_policy_client = _fake_policy_client(mocker, get_cached=mocker.Mock(return_value=([], False)))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client, policy_client=fake_policy_client)

    async def _scenario():
        slow_task = asyncio.create_task(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))
        started = time.perf_counter()
        await asyncio.sleep(0.05)  # the loop must still service this while the fetch blocks its own thread
        concurrent_elapsed = time.perf_counter() - started
        await slow_task  # let the slow request finish so no task is left dangling
        return concurrent_elapsed

    concurrent_elapsed = asyncio.run(_scenario())

    assert concurrent_elapsed < _LOOP_NOT_BLOCKED_THRESHOLD_SECONDS


def test_run_extraction_get_policies_tool_call_does_not_block_the_event_loop(monkeypatch, mocker):
    """A slow get_policies tool call doesn't block other coroutines on the event loop."""
    settings = _full_settings(monkeypatch)
    get_policies_call = {"name": "get_policies", "args": {}, "id": "call-2"}
    mock_ainvoke = mocker.AsyncMock(
        side_effect=[
            _llm_response(mocker, tool_calls=[_ACTIVATE_EXTRACT_INTENT_CALL]),
            _llm_response(mocker, tool_calls=[get_policies_call]),
            _llm_response(mocker, _ON_SCHEMA_ANSWER),
        ],
    )
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=mocker.Mock(ainvoke=mock_ainvoke)))
    fake_schema_client = _fake_schema_client(mocker, get_cached=mocker.Mock(return_value=_load_schema_fixture()))

    call_count = 0

    def _get_cached(_subject_type):
        nonlocal call_count
        call_count += 1
        if call_count > 1:  # the on-demand get_policies tool call, not run_extraction's own pre-check
            time.sleep(0.3)
        return [], False

    fake_policy_client = _fake_policy_client(mocker, get_cached=mocker.Mock(side_effect=_get_cached))
    agent = IntentAgent(settings, llm=fake_llm, schema_client=fake_schema_client, policy_client=fake_policy_client)

    async def _scenario():
        slow_task = asyncio.create_task(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))
        started = time.perf_counter()
        await asyncio.sleep(0.05)  # the loop must still service this while the tool call blocks its own thread
        concurrent_elapsed = time.perf_counter() - started
        await slow_task  # let the slow request finish so no task is left dangling
        return concurrent_elapsed

    concurrent_elapsed = asyncio.run(_scenario())

    assert concurrent_elapsed < _LOOP_NOT_BLOCKED_THRESHOLD_SECONDS


def test_schema_and_policy_fetches_use_the_dedicated_fetch_executor(monkeypatch, mocker):
    """Schema/policy fetches run on the agent's own executor, so a hung endpoint can't starve the default one."""
    settings = _full_settings(monkeypatch)
    agent = IntentAgent(settings, llm=mocker.Mock())
    calls: list[object] = []

    async def _scenario():
        loop = asyncio.get_running_loop()
        real_run_in_executor = loop.run_in_executor

        def _record(executor, func, *args):
            calls.append(executor)
            return real_run_in_executor(executor, func, *args)

        loop.run_in_executor = _record
        try:
            return await agent._run_in_fetch_executor(lambda: "ok")
        finally:
            loop.run_in_executor = real_run_in_executor

    assert asyncio.run(_scenario()) == "ok"
    assert calls == [agent._fetch_executor]
    assert agent._fetch_executor is not None  # loop's shared default executor is None, not this


def test_run_extraction_times_out_during_retry_too(monkeypatch, mocker):
    """A hanging validation retry also returns llm_timeout_error."""
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

    result = asyncio.run(agent.run_extraction("buy AAPL", RequestSubject(id="user-1", type="Customer")))

    assert json.loads(result.text) == llm_timeout_error("buy AAPL")
    assert "Stock" not in result.text
    assert mock_ainvoke.call_count == _LLM_CALLS_WITH_ONE_RETRY


def test_build_app_starts_when_schema_not_yet_populated_at_cold_start(monkeypatch, mocker, caplog):
    """An empty schema at cold start logs a WARNING, not an ERROR."""
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

    assert len(app.routes) >= _MIN_EXPECTED_ROUTES


def test_activate_skill_injects_the_schema_vocabulary_summary_not_the_raw_jgf(monkeypatch, mocker, baseline_env):
    """Activation injects the full schema summary, never the raw (truncatable) JGF."""
    baseline_env()
    monkeypatch.setenv("SCHEMA_PREVIEW_CHARS", "2000")
    schema = json.loads(_SCHEMA_FIXTURE_PATH.read_text())
    assert len(json.dumps(schema, indent=2)) > 2000  # noqa: PLR2004 - the raw form would be cut at the cap
    fake_schema_client = mocker.Mock(get_cached=mocker.Mock(return_value=schema))
    agent = IntentAgent(
        Settings(_env_file=None), llm=mocker.Mock(), schema_client=fake_schema_client, policy_client=mocker.Mock()
    )

    result = agent._skill_content("extract-intent", schema, None)

    assert "<data_schema>\n" in result
    assert "- Customer (DigitalTwin):" in result
    assert "- Customer -OWN-> Car" in result
    assert "</data_schema>" in result
    assert '"nodes"' not in result


def test_activate_skill_caps_the_injected_summary_at_schema_preview_chars(monkeypatch, mocker, baseline_env):
    baseline_env()
    monkeypatch.setenv("SCHEMA_PREVIEW_CHARS", "40")
    schema = json.loads(_SCHEMA_FIXTURE_PATH.read_text())
    fake_schema_client = mocker.Mock(get_cached=mocker.Mock(return_value=schema))
    agent = IntentAgent(
        Settings(_env_file=None), llm=mocker.Mock(), schema_client=fake_schema_client, policy_client=mocker.Mock()
    )

    result = agent._skill_content("extract-intent", schema, None)

    injected = result.split("<data_schema>\n", 1)[1].split("\n</data_schema>", 1)[0]
    assert injected == render_schema_summary(schema)[:40]


def test_run_extraction_injects_the_skill_when_the_model_answers_without_activating_it(monkeypatch, mocker, caplog):
    """An unactivated {evaluations} answer triggers skill injection and a re-ask, without spending a retry."""
    caplog.set_level("INFO")
    unbound = json.dumps(
        {
            "evaluations": [
                {
                    "subject": {"type": "Admin", "id": "attacker"},
                    "action": {"name": "RENT"},
                    "resource": {"type": "Car", "id": "kitt"},
                },
            ],
        },
    )
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [_llm_response(mocker, unbound), _llm_response(mocker, _ON_SCHEMA_ANSWER)],
    )

    result = asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    assert result.outcome == Outcome.SUCCESS
    assert result.validation_retries == 0
    assert json.loads(result.text)["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}
    assert mock_ainvoke.call_count == _LLM_CALLS_TO_FIRST_ANSWER
    second_call_messages = mock_ainvoke.call_args_list[1].args[0]
    assert "without activating the extract-intent skill" in second_call_messages[-1].content
    assert "<skill_content name='extract-intent'>" in second_call_messages[-1].content
    assert "<data_schema>" in second_call_messages[-1].content
    assert "Answered without activating extract-intent" in caplog.text


def test_run_extraction_unactivated_answer_is_then_validated_like_any_extraction(monkeypatch, mocker):
    """After the injection an off-schema answer is fed back and retried, never passed through."""
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [
            _llm_response(mocker, _OFF_SCHEMA_ANSWER),
            _llm_response(mocker, _OFF_SCHEMA_ANSWER),
            _llm_response(mocker, _ON_SCHEMA_ANSWER),
        ],
    )

    result = asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    assert result.outcome == Outcome.SUCCESS
    assert result.validation_retries == 1
    assert mock_ainvoke.call_count == _LLM_CALLS_WITH_ONE_RETRY


def test_run_extraction_unactivated_answer_then_budget_exhaustion_is_the_no_intent_error(monkeypatch, mocker):
    """After the injection, exhausting the tool budget is no_intent_error, not the placeholder."""
    get_schema_call = {"name": "get_data_schema", "args": {}, "id": "call-2"}
    agent, _ = _agent_with_responses(
        monkeypatch,
        mocker,
        [_llm_response(mocker, _ON_SCHEMA_ANSWER), _llm_response(mocker, tool_calls=[get_schema_call])],
        TOOL_CALL_MAX_ITERATIONS=1,
    )

    result = asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    assert json.loads(result.text) == no_intent_error("rent kitt")
    assert result.outcome == Outcome.ERROR


def test_run_extraction_injects_the_skill_for_an_unactivated_bare_array_answer(monkeypatch, mocker):
    unbound = json.dumps(
        [
            {
                "subject": {"type": "Admin", "id": "attacker"},
                "action": {"name": "RENT"},
                "resource": {"type": "Car", "id": "kitt"},
            }
        ],
    )
    agent, mock_ainvoke = _agent_with_responses(
        monkeypatch,
        mocker,
        [_llm_response(mocker, unbound), _llm_response(mocker, _ON_SCHEMA_ANSWER)],
    )

    result = asyncio.run(agent.run_extraction("rent kitt", RequestSubject(id="user-1", type="Customer")))

    assert result.outcome == Outcome.SUCCESS
    assert json.loads(result.text)["evaluations"][0]["subject"] == {"type": "Customer", "id": "user-1"}
    assert mock_ainvoke.call_count == _LLM_CALLS_TO_FIRST_ANSWER


def test_agent_card_advertises_the_configured_url(monkeypatch, mocker, baseline_env):
    baseline_env()
    monkeypatch.setenv("ADVERTISED_URL", "https://intent-agent.example.com/a2a")
    agent = IntentAgent(
        Settings(_env_file=None), llm=mocker.Mock(), schema_client=mocker.Mock(), policy_client=mocker.Mock()
    )

    assert agent.build_agent_card().supported_interfaces[0].url == "https://intent-agent.example.com/a2a"


@pytest.mark.parametrize(
    ("intent_port", "expected_url"),
    [
        pytest.param("6003", "http://intent:6003", id="default-port"),
        pytest.param("7000", "http://intent:7000", id="custom-port"),
    ],
)
def test_agent_card_default_url_follows_the_listening_port(
    monkeypatch, mocker, baseline_env, intent_port, expected_url
):
    baseline_env()
    monkeypatch.delenv("ADVERTISED_URL", raising=False)
    monkeypatch.setenv("INTENT_PORT", intent_port)
    agent = IntentAgent(
        Settings(_env_file=None), llm=mocker.Mock(), schema_client=mocker.Mock(), policy_client=mocker.Mock()
    )

    assert agent.build_agent_card().supported_interfaces[0].url == expected_url
