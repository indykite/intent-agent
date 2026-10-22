"""IntentExecutor audits every request exactly once and sets the task's terminal state to match."""

import asyncio
import json
import re

import pytest
from a2a import types as a2a_types
from a2a.auth.user import UnauthenticatedUser
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.routes.common import StarletteUser
from a2a.types import TaskState, TaskStatusUpdateEvent
from a2a.utils.errors import InternalError, InvalidParamsError
from starlette.authentication import SimpleUser
from starlette.requests import Request

from indykite.intent_agent.agent import ExtractionResult
from indykite.intent_agent.audit.message import Outcome
from indykite.intent_agent.config import Settings
from indykite.intent_agent.executor import IntentExecutor

_PROMPT = "rent kitt"
_EVALUATIONS = [{"action": {"name": "RENT"}, "resource": {"type": "Car", "id": "kitt"}}]
_UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_MAX_RECORDS_PER_REQUEST = 1


_DEFAULT_METADATA = object()


def _context(headers, *, metadata=_DEFAULT_METADATA, authenticated=True):
    """A RequestContext with `headers` and a one-part user message, for an authenticated caller unless
    `authenticated=False`; metadata defaults to a full subject (pass metadata=None for none).
    """
    user = StarletteUser(SimpleUser("caller-1")) if authenticated else UnauthenticatedUser()
    message = a2a_types.Message(
        role=a2a_types.Role.ROLE_USER,
        parts=[a2a_types.Part(text=_PROMPT)],
        message_id="msg-1",
        context_id="ctx-1",
        metadata=({"subject_type": "Customer", "subject_id": "user-1"} if metadata is _DEFAULT_METADATA else metadata),
    )
    return RequestContext(
        call_context=ServerCallContext(state={"headers": headers}, user=user),
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
    """Run one request through the executor and return the fake event queue."""
    event_queue = mocker.Mock(enqueue_event=mocker.AsyncMock())
    asyncio.run(executor.execute(context, event_queue))
    return event_queue


def _status_events(event_queue):
    """Every TaskStatusUpdateEvent's Status enqueued, in order."""
    return [
        call.args[0].status
        for call in event_queue.enqueue_event.call_args_list
        if isinstance(call.args[0], TaskStatusUpdateEvent)
    ]


def _terminal_task_state(event_queue):
    """The State of the last TaskStatusUpdateEvent enqueued."""
    return _status_events(event_queue)[-1].state


def _history_summary_message(event_queue):
    """The summary text on the last WORKING status, which a2a archives into history before the terminal one."""
    statuses = _status_events(event_queue)
    assert not statuses[-1].HasField("message"), "the final status must be message-free"
    return statuses[-2].message.parts[0].text


def _sole_record(sender):
    """The one record the request produced, asserting there is exactly one."""
    assert sender.send.await_count == _MAX_RECORDS_PER_REQUEST
    return sender.send.await_args.args[0]


def test_success_records_the_extraction(baseline_env, mocker):
    """The record carries the pipeline's classification, the subject, the ids and what the caller was sent."""
    result = ExtractionResult(
        text="{...}",
        outcome=Outcome.SUCCESS,
        evaluations=_EVALUATIONS,
        validation_retries=1,
        schema_version="deadbeef",
        policy_version="cafef00d",
        policy_fetch_failed=False,
    )
    executor, sender, _agent = _executor(baseline_env, mocker, result=result)

    event_queue = _execute(
        executor, _context({"authorization": "Bearer good-token", "x-request-id": "req-123"}), mocker
    )

    assert _terminal_task_state(event_queue) == TaskState.TASK_STATE_COMPLETED
    assert _history_summary_message(event_queue) == "Intent extracted successfully."
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
    assert record.policy_version == "cafef00d"
    assert record.policy_fetch_failed is False
    assert record.llm_provider == "gemini"
    assert record.llm_model == "gemini-3.5-flash-lite"
    assert record.validation_retries == 1
    assert record.message == ""
    assert record.latency_ms >= 0


def test_audit_ordering_around_the_terminal_task_event(baseline_env, mocker):
    """The audit record is written before the terminal event, and a failed enqueue doesn't add a second one."""
    success_result = ExtractionResult(text="{...}", outcome=Outcome.SUCCESS, evaluations=_EVALUATIONS)

    executor, sender, _agent = _executor(baseline_env, mocker, result=success_result)
    calls = []
    sender.send.side_effect = lambda *_args, **_kwargs: calls.append("audit")

    async def _record_terminal_event(event):
        if isinstance(event, TaskStatusUpdateEvent) and event.status.state in (
            TaskState.TASK_STATE_COMPLETED,
            TaskState.TASK_STATE_FAILED,
        ):
            calls.append("terminal_event")

    event_queue = mocker.Mock(enqueue_event=mocker.AsyncMock(side_effect=_record_terminal_event))
    asyncio.run(executor.execute(_context({"authorization": "Bearer good-token"}), event_queue))
    assert calls == ["audit", "terminal_event"]

    executor, sender, _agent = _executor(baseline_env, mocker, result=success_result)
    status_events_seen = 0
    post_audit_status_event = 2  # the WORKING update _emit_terminal_status enqueues first
    queue_closed = RuntimeError("queue closed")

    async def _fail_the_status_event_right_after_the_audit_call(event):
        nonlocal status_events_seen
        if isinstance(event, TaskStatusUpdateEvent):
            status_events_seen += 1
            if status_events_seen == post_audit_status_event:
                raise queue_closed

    event_queue = mocker.Mock(
        enqueue_event=mocker.AsyncMock(side_effect=_fail_the_status_event_right_after_the_audit_call)
    )
    with pytest.raises(RuntimeError, match="queue closed"):
        asyncio.run(executor.execute(_context({"authorization": "Bearer good-token"}), event_queue))
    assert sender.send.await_count == _MAX_RECORDS_PER_REQUEST


@pytest.mark.parametrize(
    "outcome",
    [Outcome.VALIDATION_FAILED, Outcome.UNMATCHED, Outcome.ERROR],
    ids=["validation_failed", "unmatched", "error"],
)
def test_non_success_outcomes_surface_as_task_state_failed(baseline_env, mocker, outcome):
    """Every non-success outcome ends the task as TASK_STATE_FAILED with a fixed summary message."""
    executor, _sender, _agent = _executor(
        baseline_env, mocker, result=ExtractionResult(text="{...}", outcome=outcome, message="did not work out")
    )

    event_queue = _execute(executor, _context({"authorization": "Bearer good-token"}), mocker)

    assert _terminal_task_state(event_queue) == TaskState.TASK_STATE_FAILED
    assert _history_summary_message(event_queue) == "Failed to extract intent."


def test_policy_fetch_failure_is_recorded_on_the_audit_trail(baseline_env, mocker):
    """A failed policy fetch is recorded on the audit record even when the request succeeds."""
    result = ExtractionResult(
        text="{...}",
        outcome=Outcome.SUCCESS,
        evaluations=_EVALUATIONS,
        policy_version="",
        policy_fetch_failed=True,
    )
    executor, sender, _agent = _executor(baseline_env, mocker, result=result)

    _execute(executor, _context({"authorization": "Bearer good-token"}), mocker)

    record = _sole_record(sender)
    assert record.policy_version == ""
    assert record.policy_fetch_failed is True


def test_generates_a_request_id_when_the_caller_sends_none(baseline_env, mocker):
    """No X-Request-ID means a server-generated UUID."""
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


def test_unauthenticated_call_is_audited_as_error_and_still_rejected(baseline_env, mocker):
    """An unauthenticated call reaching the executor is InternalError, audited with the same message."""
    executor, sender, agent = _executor(baseline_env, mocker)

    with pytest.raises(InternalError) as excinfo:
        _execute(executor, _context({"x-request-id": "req-401"}, authenticated=False), mocker)

    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == excinfo.value.message == "Request reached the executor unauthenticated"
    assert record.request_id == "req-401"
    assert record.subject_id == ""
    assert record.prompt == _PROMPT
    assert record.evaluations == []
    agent.run_extraction.assert_not_awaited()


def test_missing_subject_type_is_audited_as_error_and_rejected(baseline_env, mocker):
    """A missing subject_type is InvalidParamsError, audited as `error` with its message."""
    executor, sender, agent = _executor(baseline_env, mocker)

    context = _context({"authorization": "Bearer good-token"}, metadata={"subject_id": "user-1"})
    with pytest.raises(InvalidParamsError) as excinfo:
        _execute(executor, context, mocker)

    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == excinfo.value.message == "subject_type is required"
    assert record.subject_id == ""
    agent.run_extraction.assert_not_awaited()


def test_missing_subject_id_is_audited_as_error_and_rejected(baseline_env, mocker):
    """A missing subject_id is InvalidParamsError, audited like a missing subject_type."""
    executor, sender, agent = _executor(baseline_env, mocker)

    context = _context({"authorization": "Bearer good-token"}, metadata={"subject_type": "Customer"})
    with pytest.raises(InvalidParamsError) as excinfo:
        _execute(executor, context, mocker)

    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == excinfo.value.message == "subject_id is required"
    agent.run_extraction.assert_not_awaited()


def test_whitespace_only_subject_type_is_audited_as_error_and_rejected(baseline_env, mocker):
    """A whitespace-only subject_type counts as missing."""
    executor, sender, agent = _executor(baseline_env, mocker)

    context = _context({"authorization": "Bearer good-token"}, metadata={"subject_type": "   "})
    with pytest.raises(InvalidParamsError) as excinfo:
        _execute(executor, context, mocker)

    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == excinfo.value.message == "subject_type is required"
    agent.run_extraction.assert_not_awaited()


@pytest.mark.parametrize("key", ["subject_type", "subject_id"])
def test_non_string_metadata_value_is_audited_as_error_and_rejected(baseline_env, mocker, key):
    """A non-string subject_type/subject_id is rejected rather than str()-coerced."""
    executor, sender, agent = _executor(baseline_env, mocker)

    context = _context(
        {"authorization": "Bearer good-token"},
        metadata={"subject_type": "Customer", "subject_id": "user-1", key: {"nested": "object"}},
    )
    with pytest.raises(InvalidParamsError) as excinfo:
        _execute(executor, context, mocker)

    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == excinfo.value.message == f"{key} must be a string"
    agent.run_extraction.assert_not_awaited()


def test_missing_metadata_object_is_audited_as_error_and_rejected(baseline_env, mocker):
    """A message with no metadata object is the same InvalidParamsError as a missing subject_type."""
    executor, sender, agent = _executor(baseline_env, mocker)

    context = _context({"authorization": "Bearer good-token"}, metadata=None)
    with pytest.raises(InvalidParamsError) as excinfo:
        _execute(executor, context, mocker)

    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == excinfo.value.message == "subject_type is required"
    agent.run_extraction.assert_not_awaited()


def _request(body: bytes, headers: dict[str, str]) -> Request:
    """A Starlette request with this body and these headers, as the middleware hands it over."""

    async def _receive():
        return {"type": "http.request", "body": body, "more_body": False}

    raw_headers = [(name.lower().encode(), value.encode()) for name, value in headers.items()]
    return Request({"type": "http", "method": "POST", "path": "/", "headers": raw_headers}, _receive)


def _send_message_body(*, context_id: str | None = None) -> bytes:
    """A SendMessage JSON-RPC body carrying the test prompt."""
    message = {"role": "ROLE_USER", "parts": [{"text": _PROMPT}], "messageId": "msg-1"}
    if context_id:
        message["contextId"] = context_id
    return json.dumps({"jsonrpc": "2.0", "id": "1", "method": "SendMessage", "params": {"message": message}}).encode()


def test_rejected_send_message_is_audited_with_its_prompt(baseline_env, mocker):
    """A refused SendMessage is audited with its reason, request id, prompt and context id."""
    executor, sender, agent = _executor(baseline_env, mocker)
    request = _request(_send_message_body(context_id="ctx-9"), {"x-request-id": "req-401"})

    asyncio.run(executor.audit_rejected_request(request, "Token is not active"))

    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == "Token is not active"
    assert record.request_id == "req-401"
    assert record.prompt == _PROMPT
    assert record.context_id == "ctx-9"
    assert record.subject_id == ""
    assert record.evaluations == []
    agent.run_extraction.assert_not_awaited()


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({"jsonrpc": "2.0", "id": "1", "method": "ListTasks", "params": {}}).encode(),
        b"not json",
        b"[]",
        json.dumps({"jsonrpc": "2.0", "id": "1", "method": "SendMessage", "params": {"message": 7}}).encode(),
    ],
    ids=["other-method", "not-json", "not-an-object", "malformed-message"],
)
def test_rejected_request_without_a_readable_message_is_still_audited(baseline_env, mocker, body):
    """Any other refused request is still audited, with an empty prompt."""
    executor, sender, _agent = _executor(baseline_env, mocker)

    asyncio.run(executor.audit_rejected_request(_request(body, {}), "Authorization required"))

    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == "Authorization required"
    assert record.prompt == ""
    assert record.context_id == ""


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
    """llm_provider is the family prefix of the configured GEMINI_MODEL."""
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
    """A raising audit sender doesn't fail a successful request; the failure is logged."""
    caplog.set_level("ERROR")
    executor, sender, _agent = _executor(
        baseline_env, mocker, result=ExtractionResult(text="{...}", outcome=Outcome.SUCCESS)
    )
    sender.send.side_effect = RuntimeError("audit backend down")

    _execute(executor, _context({"authorization": "Bearer good-token", "x-request-id": "req-ok"}), mocker)

    assert sender.send.await_count == _MAX_RECORDS_PER_REQUEST
    assert "Failed to audit request req-ok (outcome success)" in caplog.text
    assert "audit backend down" in caplog.text


def test_audit_failure_does_not_override_the_rejection(baseline_env, mocker, caplog):
    """A raising audit sender doesn't replace the rejection; the failure is logged."""
    caplog.set_level("ERROR")
    executor, sender, _agent = _executor(baseline_env, mocker)
    sender.send.side_effect = RuntimeError("audit backend down")

    with pytest.raises(InternalError) as excinfo:
        _execute(executor, _context({"x-request-id": "req-401"}, authenticated=False), mocker)

    assert excinfo.value.message == "Request reached the executor unauthenticated"
    assert "Failed to audit request req-401 (outcome error)" in caplog.text


def test_unexpected_failure_is_audited_and_re_raised(baseline_env, mocker):
    """A pipeline crash is audited as `error` and still propagates."""
    executor, sender, _agent = _executor(
        baseline_env, mocker, run_extraction=mocker.AsyncMock(side_effect=RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        _execute(executor, _context({"authorization": "Bearer good-token"}), mocker)

    record = _sole_record(sender)
    assert record.outcome == Outcome.ERROR
    assert record.message == "RuntimeError: boom"
    assert record.subject_id == "user-1"
