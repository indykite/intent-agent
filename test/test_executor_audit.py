"""IntentExecutor emits exactly one TDD 4.9 audit record per request - on success, on a rejected
token (like the IAG audits every rejection), and on an unexpected failure.
"""

import asyncio
import re

import pytest
from a2a import types as a2a_types
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from starlette.exceptions import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED

from indykite.intent_agent import auth
from indykite.intent_agent.agent import ExtractionResult
from indykite.intent_agent.audit.message import Outcome
from indykite.intent_agent.config import Settings
from indykite.intent_agent.executor import IntentExecutor

_PROMPT = "rent kitt"
_EVALUATIONS = [{"action": {"name": "RENT"}, "resource": {"type": "Car", "id": "kitt"}}]
_UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_MAX_RECORDS_PER_REQUEST = 1


_DEFAULT_METADATA = object()


def _context(headers, *, metadata=_DEFAULT_METADATA):
    """A RequestContext carrying `headers` and a one-part user message, as the A2A routes
    build it. subject_type/subject_id now come from the message's metadata (the caller
    resolves them, not a token) - defaults to a full pair so most tests don't need to specify
    it; a test exercising the metadata itself (e.g. a missing subject_type, or no metadata
    object at all - pass metadata=None for that) passes its own.
    """
    message = a2a_types.Message(
        role=a2a_types.Role.ROLE_USER,
        parts=[a2a_types.Part(text=_PROMPT)],
        message_id="msg-1",
        context_id="ctx-1",
        metadata=({"subject_type": "Customer", "subject_id": "user-1"} if metadata is _DEFAULT_METADATA else metadata),
    )
    return RequestContext(
        call_context=ServerCallContext(state={"headers": headers}),
        request=a2a_types.SendMessageRequest(message=message),
        context_id="ctx-1",
    )


def _executor(baseline_env, mocker, *, result=None, run_extraction=None):
    """An IntentExecutor over a fake agent and a recording audit sender, plus both fakes."""
    baseline_env()
    settings = Settings(_env_file=None)
    extraction = run_extraction or mocker.AsyncMock(return_value=result)
    agent = mocker.Mock(settings=settings, run_extraction=extraction)
    sender = mocker.Mock(send=mocker.AsyncMock(), close=mocker.AsyncMock())
    return IntentExecutor(agent, sender), sender, agent


def _execute(executor, context, mocker):
    """Run one request through the executor with a fake event queue."""
    event_queue = mocker.Mock(enqueue_event=mocker.AsyncMock())
    asyncio.run(executor.execute(context, event_queue))


def _sole_record(sender):
    """The one record the request produced - exactly one, per TDD 4.9."""
    assert sender.send.await_count == _MAX_RECORDS_PER_REQUEST
    return sender.send.await_args.args[0]


def test_success_records_the_extraction(baseline_env, mocker):
    """The terminal record carries the pipeline's classification, the token's subject, the A2A
    context id, the caller's request id, and what the caller was sent.
    """
    mocker.patch.object(auth, "validate_access_token", mocker.AsyncMock(return_value={"active": True, "sub": "user-1"}))
    result = ExtractionResult(
        text="{...}",
        outcome=Outcome.SUCCESS,
        evaluations=_EVALUATIONS,
        validation_retries=1,
        schema_version="deadbeef",
    )
    executor, sender, _agent = _executor(baseline_env, mocker, result=result)

    _execute(executor, _context({"authorization": "Bearer good-token", "x-request-id": "req-123"}), mocker)

    record = _sole_record(sender)
    assert record.outcome == Outcome.SUCCESS
    assert record.request_id == "req-123"
    assert record.context_id == "ctx-1"
    assert record.subject_id == "user-1"
    assert record.subject_type == "Customer"
    assert record.requesting_agent_id == ""
    assert record.prompt == _PROMPT
    assert record.evaluations == _EVALUATIONS
    assert record.schema_version == "deadbeef"
    assert record.llm_provider == "gemini"
    assert record.llm_model == "gemini-3.5-flash-lite"
    assert record.validation_retries == 1
    assert record.message == ""
    assert record.latency_ms >= 0


def test_generates_a_request_id_when_the_caller_sends_none(baseline_env, mocker):
    """No X-Request-ID means a server-generated UUID (TDD: `request_id` (UUID), server)."""
    mocker.patch.object(auth, "validate_access_token", mocker.AsyncMock(return_value={"active": True, "sub": "user-1"}))
    executor, sender, _agent = _executor(
        baseline_env,
        mocker,
        result=ExtractionResult(text="", outcome=Outcome.UNMATCHED, message="nothing matched"),
    )

    _execute(executor, _context({"authorization": "Bearer good-token"}), mocker)

    record = _sole_record(sender)
    assert _UUID_PATTERN.match(record.request_id)
    assert record.outcome == Outcome.UNMATCHED
    assert record.message == "nothing matched"


def test_missing_bearer_is_audited_as_error_and_still_rejected(baseline_env, mocker):
    """A request with no token is rejected with 401 AND audited (IAG: "no bearer token") -
    as `outcome = error` with the same reason the caller gets (TDD owner's decision).
    """
    executor, sender, agent = _executor(baseline_env, mocker)

    with pytest.raises(HTTPException) as excinfo:
        _execute(executor, _context({"x-request-id": "req-401"}), mocker)

    assert excinfo.value.status_code == HTTP_401_UNAUTHORIZED
    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == excinfo.value.detail == "Authorization required"
    assert record.request_id == "req-401"
    assert record.subject_id == ""
    assert record.prompt == _PROMPT
    assert record.evaluations == []
    agent.run_extraction.assert_not_awaited()


def test_missing_subject_type_is_audited_as_error_and_rejected(baseline_env, mocker):
    """A caller-supplied subject_type is required (IAG resolves and supplies it, not the
    token) - missing/blank it is a 400, audited the same way a rejected token is.
    """
    mocker.patch.object(auth, "validate_access_token", mocker.AsyncMock(return_value={"active": True}))
    executor, sender, agent = _executor(baseline_env, mocker)

    context = _context({"authorization": "Bearer good-token"}, metadata={"subject_id": "user-1"})
    with pytest.raises(HTTPException) as excinfo:
        _execute(executor, context, mocker)

    assert excinfo.value.status_code == HTTP_400_BAD_REQUEST
    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == excinfo.value.detail == "subject_type is required"
    assert record.subject_id == ""
    agent.run_extraction.assert_not_awaited()


def test_whitespace_only_subject_type_is_audited_as_error_and_rejected(baseline_env, mocker):
    """A whitespace-only subject_type must count as blank - the same 400 as a missing one, not
    a value that happens to be truthy and slips past the check.
    """
    mocker.patch.object(auth, "validate_access_token", mocker.AsyncMock(return_value={"active": True}))
    executor, sender, agent = _executor(baseline_env, mocker)

    context = _context({"authorization": "Bearer good-token"}, metadata={"subject_type": "   "})
    with pytest.raises(HTTPException) as excinfo:
        _execute(executor, context, mocker)

    assert excinfo.value.status_code == HTTP_400_BAD_REQUEST
    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == excinfo.value.detail == "subject_type is required"
    agent.run_extraction.assert_not_awaited()


@pytest.mark.parametrize("key", ["subject_type", "subject_id"])
def test_non_string_metadata_value_is_audited_as_error_and_rejected(baseline_env, mocker, key):
    """A non-string subject_type/subject_id (e.g. a nested object) must be rejected outright,
    not silently str()-coerced into a Python repr and bound into every evaluation/audit record.
    """
    mocker.patch.object(auth, "validate_access_token", mocker.AsyncMock(return_value={"active": True}))
    executor, sender, agent = _executor(baseline_env, mocker)

    context = _context(
        {"authorization": "Bearer good-token"},
        metadata={"subject_type": "Customer", "subject_id": "user-1", key: {"nested": "object"}},
    )
    with pytest.raises(HTTPException) as excinfo:
        _execute(executor, context, mocker)

    assert excinfo.value.status_code == HTTP_400_BAD_REQUEST
    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == excinfo.value.detail == f"{key} must be a string"
    agent.run_extraction.assert_not_awaited()


def test_missing_metadata_object_is_audited_as_error_and_rejected(baseline_env, mocker):
    """A message with no metadata object at all (metadata=None, not just an empty/incomplete
    one) must be the same 400 as a missing subject_type, not an unhandled TypeError from
    treating None as a mapping.
    """
    mocker.patch.object(auth, "validate_access_token", mocker.AsyncMock(return_value={"active": True}))
    executor, sender, agent = _executor(baseline_env, mocker)

    context = _context({"authorization": "Bearer good-token"}, metadata=None)
    with pytest.raises(HTTPException) as excinfo:
        _execute(executor, context, mocker)

    assert excinfo.value.status_code == HTTP_400_BAD_REQUEST
    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == excinfo.value.detail == "subject_type is required"
    agent.run_extraction.assert_not_awaited()


def test_rejected_token_is_audited_with_the_introspection_reason(baseline_env, mocker):
    """An inactive/invalid token is audited with the exact reason handed back in the 401."""
    mocker.patch.object(
        auth,
        "validate_access_token",
        mocker.AsyncMock(side_effect=auth.TokenValidationError("Token is not active")),
    )
    executor, sender, _agent = _executor(baseline_env, mocker)

    with pytest.raises(HTTPException) as excinfo:
        _execute(executor, _context({"authorization": "Bearer bad-token"}), mocker)

    assert excinfo.value.detail == "Token is not active"
    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == "Token is not active"
    assert record.subject_id == ""


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("gemini-3.5-flash-lite", "gemini"),
        ("gemini-3.1-flash-lite", "gemini"),
        ("models/gemini-3.5-flash-lite", "gemini"),
        ("claude-sonnet-5", "claude"),
    ],
)
def test_llm_provider_is_derived_from_the_configured_model(baseline_env, monkeypatch, mocker, model, provider):
    """llm_provider comes from GEMINI_MODEL, not a hard-coded constant, so the record can't
    disagree with what is configured - the family prefix of the model name.
    """
    mocker.patch.object(auth, "validate_access_token", mocker.AsyncMock(return_value={"active": True, "sub": "user-1"}))
    baseline_env()
    monkeypatch.setenv("GEMINI_MODEL", model)
    executor, sender, _agent = _executor(
        baseline_env, mocker, result=ExtractionResult(text="{...}", outcome=Outcome.SUCCESS)
    )

    _execute(executor, _context({"authorization": "Bearer good-token"}), mocker)

    record = _sole_record(sender)
    assert record.llm_provider == provider
    assert record.llm_model == model


def test_audit_failure_does_not_change_a_successful_request(baseline_env, mocker, caplog):
    """A sender that blows up (against its own contract) must not turn a request whose artifact
    the caller already has into an error - it is logged with the request id instead.
    """
    caplog.set_level("ERROR")
    mocker.patch.object(auth, "validate_access_token", mocker.AsyncMock(return_value={"active": True, "sub": "user-1"}))
    executor, sender, _agent = _executor(
        baseline_env, mocker, result=ExtractionResult(text="{...}", outcome=Outcome.SUCCESS)
    )
    sender.send.side_effect = RuntimeError("audit backend down")

    _execute(executor, _context({"authorization": "Bearer good-token", "x-request-id": "req-ok"}), mocker)

    assert sender.send.await_count == _MAX_RECORDS_PER_REQUEST
    assert "Failed to audit request req-ok (outcome success)" in caplog.text
    assert "audit backend down" in caplog.text


def test_audit_failure_does_not_override_the_rejection(baseline_env, mocker, caplog):
    """A sender failure while auditing a 401 must not replace that 401 with the sender's own
    exception - the caller still gets the rejection, and the audit failure is logged.
    """
    caplog.set_level("ERROR")
    executor, sender, _agent = _executor(baseline_env, mocker)
    sender.send.side_effect = RuntimeError("audit backend down")

    with pytest.raises(HTTPException) as excinfo:
        _execute(executor, _context({"x-request-id": "req-401"}), mocker)

    assert excinfo.value.status_code == HTTP_401_UNAUTHORIZED
    assert excinfo.value.detail == "Authorization required"
    assert "Failed to audit request req-401 (outcome error)" in caplog.text


def test_unexpected_failure_is_audited_and_re_raised(baseline_env, mocker):
    """A crash inside the pipeline still leaves one record behind (outcome=error, the exception
    as the message) and still propagates - auditing must never swallow the failure.
    """
    mocker.patch.object(auth, "validate_access_token", mocker.AsyncMock(return_value={"active": True, "sub": "user-1"}))
    executor, sender, _agent = _executor(
        baseline_env, mocker, run_extraction=mocker.AsyncMock(side_effect=RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        _execute(executor, _context({"authorization": "Bearer good-token"}), mocker)

    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == "RuntimeError: boom"
    assert record.subject_id == "user-1"
