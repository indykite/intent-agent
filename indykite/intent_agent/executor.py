"""A2A executor: runs the extraction, emits the task events, and audits every request exactly
once (TDD 4.9).
"""

import json
import logging
import time
from collections.abc import Mapping
from typing import Any

from a2a.helpers import new_task_from_user_message, new_text_artifact, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.types import (
    Message,
    SendMessageRequest,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils.errors import A2AError, InternalError, InvalidParamsError
from google.protobuf.json_format import ParseDict, ParseError
from starlette.requests import Request

from indykite.intent_agent import auth
from indykite.intent_agent.agent import ExtractionResult, IntentAgent, RequestSubject
from indykite.intent_agent.audit.message import AuditMessage, Outcome, request_id_from_headers, utc_timestamp
from indykite.intent_agent.audit.sender import Sender
from indykite.intent_agent.config import Settings

logger = logging.getLogger(__name__)

_MILLISECONDS_PER_SECOND = 1000
_EXTRACTION_SUCCEEDED_MESSAGE = "Intent extracted successfully."
_EXTRACTION_FAILED_MESSAGE = "Failed to extract intent."
_UNAUTHENTICATED_CALL_MESSAGE = "Request reached the executor unauthenticated"


def _prompt_from_message(message: Message | None) -> str:
    """Concatenate the text parts of an A2A message."""
    raw_text = ""
    if message:
        for part in message.parts or []:
            text = getattr(part, "text", None)
            if text:
                raw_text += text
    return raw_text


def _message_from_body(body: bytes) -> Message | None:
    """The message of a SendMessage JSON-RPC body; None for anything else."""
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict) or payload.get("method") != "SendMessage":
            return None
        request = ParseDict(payload.get("params") or {}, SendMessageRequest(), ignore_unknown_fields=True)
    except (ValueError, ParseError):
        return None
    return request.message if request.HasField("message") else None


def subject_from_message(message: Message | None) -> tuple[str, str]:
    """Read subject_type/subject_id from the caller-supplied message metadata - IAG
    resolves the caller's own identity itself and supplies both per request, instead of
    subject_id coming from a token's `sub` claim and subject_type from static config.

    Both are required (ENG-9810) - InvalidParamsError (JSON-RPC -32602) if either is missing or blank
    (whitespace-only counts as blank). subject_id used to be reliably populated from the
    token's own `sub` claim, so a real authenticated caller was never actually anonymous in
    practice; now that IAG must supply it explicitly instead, a missing subject_id is a
    caller-side bug (it forgot to resolve/attach the caller's identity), not a legitimate
    anonymous-caller request, and must be rejected the same as a missing subject_type rather
    than silently proceeding with an unidentified caller. Either present but not a string
    (e.g. a nested object) is also rejected - str(some_dict) would otherwise silently coerce it
    into a Python repr and bind that into every evaluation and audit record, instead of
    surfacing the malformed request to the caller.
    """
    metadata = dict(message.metadata) if message and message.metadata else {}
    subject_type = _require_metadata_string(metadata, "subject_type")
    subject_id = _require_metadata_string(metadata, "subject_id")
    if not subject_type:
        raise InvalidParamsError(message="subject_type is required")
    if not subject_id:
        raise InvalidParamsError(message="subject_id is required")
    return subject_type, subject_id


def _require_metadata_string(metadata: dict[str, Any], key: str) -> str:
    """Return metadata[key] as a stripped string, or "" if absent/blank - raising
    InvalidParamsError if it's present but not a string at all (see subject_from_message).
    """
    value = metadata.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise InvalidParamsError(message=f"{key} must be a string")
    return value.strip()


def _llm_provider(model: str) -> str:
    """The provider family of a configured model name: the segment before the first `-` of
    its last path component, so "gemini-3.5-flash-lite" (or "models/gemini-3.5-flash-lite")
    audits as "gemini". Derived from GEMINI_MODEL rather than hard-coded, so the record can't
    disagree with what is actually configured.
    """
    return model.rsplit("/", 1)[-1].split("-", 1)[0]


class _RequestAudit:
    """Accumulates the per-request audit fields as the request progresses, the way the IAG's
    token service builds up its `attempt` before sending - so every exit path (rejected
    token, extraction, unexpected failure) turns into one record through the same
    `record()`. Every request produces exactly one record (TDD 4.9).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        headers: Mapping[str, str],
        context_id: str,
        prompt: str,
        started: float | None = None,
    ) -> None:
        """Capture what's known before the request is handled: ids, prompt, and the start time
        (auth.REQUEST_STARTED_KEY when the middleware set it, so latency includes authentication).
        """
        self._settings = settings
        self._started = time.perf_counter() if started is None else started
        self.request_id = request_id_from_headers(headers)
        self.context_id = context_id
        self.prompt = prompt
        self.subject_id = ""
        self.subject_type = ""

    @classmethod
    def for_context(cls, settings: Settings, context: RequestContext) -> "_RequestAudit":
        """The audit of a request that reached the executor."""
        state = context.call_context.state if context.call_context else {}
        return cls(
            settings,
            headers=state.get("headers") or {},
            context_id=context.context_id or "",
            prompt=_prompt_from_message(context.message),
            started=state.get(auth.REQUEST_STARTED_KEY),
        )

    def record(self, result: ExtractionResult) -> AuditMessage:
        """Build the audit record for this request from what the pipeline produced."""
        latency_ms = round((time.perf_counter() - self._started) * _MILLISECONDS_PER_SECOND)
        return AuditMessage(
            timestamp=utc_timestamp(),
            request_id=self.request_id,
            subject_id=self.subject_id,
            subject_type=self.subject_type,
            # v1.1: the ik_token actor claim (see auth.validate_ik_token) - no ITS yet.
            requesting_agent_id="",
            context_id=self.context_id,
            prompt=self.prompt,
            evaluations=result.evaluations,
            schema_version=result.schema_version,
            policy_version=result.policy_version,
            llm_provider=_llm_provider(self._settings.gemini_model),
            llm_model=self._settings.gemini_model,
            latency_ms=latency_ms,
            validation_retries=result.validation_retries,
            policy_fetch_failed=result.policy_fetch_failed,
            outcome=result.outcome,
            message=result.message,
        )


class IntentExecutor(AgentExecutor):
    """AgentExecutor that extracts intent from incoming text using an LLM."""

    def __init__(self, agent: IntentAgent, audit_sender: Sender) -> None:
        """Wrap the given IntentAgent as an A2A AgentExecutor, auditing every request."""
        self._agent = agent
        self._audit_sender = audit_sender

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle one A2A task: extract intent, emit the result artifact, audit the outcome.

        A rejected request is audited too, like the IAG audits every rejection - as
        `outcome = error` with the same reason the caller gets (TDD owner's decision) - and so
        is an unexpected exception, so a request can't slip through unrecorded. Auditing never
        changes the request's own outcome: see _audit.

        Exactly one of the three `_audit` calls below ever runs for a given request, by
        construction rather than by a guard flag: the `try` block's own two `except` clauses
        audit-and-reraise immediately, so nothing past this method's `try` statement can also
        run when one of them fires; the line auditing `result` is reachable only once the `try`
        block has run to completion without raising, at which point neither `except` clause can
        fire for it any more. `_emit_terminal_status`, called after that line, is deliberately
        outside the `try` - a failure there propagates straight out instead of through either
        `except`, so it can never trigger a second audit for a request already recorded as a
        success. That is also why the audit itself has to happen between `_run_extraction` and
        `_emit_terminal_status`, not after both: see `_emit_terminal_status`'s docstring.
        """
        audit_record = _RequestAudit.for_context(self._agent.settings, context)
        try:
            self._authenticate(context)
            audit_record.subject_type, audit_record.subject_id = self._subject_params(context)
            result = await self._run_extraction(context, event_queue, audit_record)
        except A2AError as exc:
            # The SDK answers it as a JSON-RPC error carrying exc.message - audit that same text.
            await self._audit(audit_record, ExtractionResult(text="", outcome=Outcome.ERROR, message=exc.message))
            raise
        except Exception as exc:
            await self._audit(
                audit_record,
                ExtractionResult(text="", outcome=Outcome.ERROR, message=f"{type(exc).__name__}: {exc}"),
            )
            raise

        await self._audit(audit_record, result)
        await self._emit_terminal_status(context, event_queue, result)

    async def _audit(self, audit_record: _RequestAudit, result: ExtractionResult) -> None:
        """Record the request's outcome; never raises.

        The sender contract is already "log, don't raise", but building the record or a
        misbehaving sender could still fail - and that must not override what the request
        itself produced: the A2A error / exception being re-raised in `execute`, or a result whose
        artifact the caller already has. An audit failure is logged with the request id instead.
        """
        try:
            await self._audit_sender.send(audit_record.record(result))
        except Exception:
            logger.exception("Failed to audit request %s (outcome %s)", audit_record.request_id, result.outcome)

    async def audit_rejected_request(self, request: Request, message: str) -> None:
        """Audit a request auth.BearerAuthMiddleware refused; never raises."""
        started = getattr(request.state, auth.REQUEST_STARTED_KEY, None)
        try:
            body_message = _message_from_body(await request.body())
            audit_record = _RequestAudit(
                self._agent.settings,
                headers=request.headers,
                context_id=body_message.context_id if body_message else "",
                prompt=_prompt_from_message(body_message),
                started=started,
            )
        except Exception:
            logger.exception("Failed to read the rejected request for its audit record")
            audit_record = _RequestAudit(
                self._agent.settings, headers=request.headers, context_id="", prompt="", started=started
            )
        await self._audit(audit_record, ExtractionResult(text="", outcome=Outcome.ERROR, message=message))

    async def audit_refused_message(self, params: SendMessageRequest, context: ServerCallContext, message: str) -> None:
        """Audit a SendMessage the request handler refused before it reached `execute`."""
        audit_record = _RequestAudit(
            self._agent.settings,
            headers=context.state.get("headers") or {},
            context_id=params.message.context_id,
            prompt=_prompt_from_message(params.message),
            started=context.state.get(auth.REQUEST_STARTED_KEY),
        )
        await self._audit(audit_record, ExtractionResult(text="", outcome=Outcome.ERROR, message=message))

    @staticmethod
    def _authenticate(context: RequestContext) -> None:
        """Refuse an unauthenticated call - fails closed if wired without the middleware. Reaching
        this is a wiring bug, not the caller's, so it is InternalError (JSON-RPC -32603).
        """
        call_context = context.call_context
        if not call_context or not call_context.user.is_authenticated:
            raise InternalError(message=_UNAUTHENTICATED_CALL_MESSAGE)

    @staticmethod
    def _subject_params(context: RequestContext) -> tuple[str, str]:
        """subject_type/subject_id of the request (subject_from_message). The request handler
        already refused invalid ones before the SDK started a task; this is the second layer.
        """
        return subject_from_message(context.message)

    async def _run_extraction(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        audit_record: _RequestAudit,
    ) -> ExtractionResult:
        """Run the extraction for an authenticated request, emitting the task/artifact events.

        Deliberately stops short of the terminal status event - `execute` audits the result
        returned here first, then calls `_emit_terminal_status` (see that docstring for why
        the terminal event has to come after the audit, not before).
        """
        logger.info(
            "Received message for %s: %s",
            self._agent.settings.intent_agent_name,
            audit_record.prompt,
        )

        prompt = audit_record.prompt or "(empty)"

        task = context.current_task or new_task_from_user_message(context.message)
        await event_queue.enqueue_event(task)

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.TASK_STATE_WORKING,
                    message=new_text_message("Extracting intent..."),
                ),
            ),
        )

        result = await self._agent.run_extraction(
            prompt, RequestSubject(id=audit_record.subject_id, type=audit_record.subject_type)
        )

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                artifact=new_text_artifact(name="result", text=result.text),
            ),
        )
        return result

    async def _emit_terminal_status(
        self, context: RequestContext, event_queue: EventQueue, result: ExtractionResult
    ) -> None:
        """Emit the task's terminal status, after the request has already been audited.

        The terminal TaskState reflects the outcome (ENG-9804): TASK_STATE_COMPLETED only for
        a successful extraction, TASK_STATE_FAILED for every other outcome (validation failure,
        no-intent/unmatched, timeout, schema unavailable, ...) - the caller's artifact JSON
        still carries the fixed {prompt, error, reason, obligation} shape (errors.py) either way,
        this only changes what the task's own status reports.

        A fixed summary message (the detailed reason stays in the artifact JSON's `reason`
        field, not duplicated here) must land in the task's `history`, not just sit on the
        terminal status - a2a's TaskManager only archives a status's message into history once
        a *later* TaskStatusUpdateEvent supersedes it. So the message is carried on one last
        WORKING update first, replacing the pre-extraction "Extracting intent..." placeholder as
        the final word instead of leaving it there, and the truly final status is left
        message-free - never the terminal one itself: DefaultRequestHandler.on_message_send
        returns on the *first* terminal-state Task snapshot it sees, so a message on that event
        would never get superseded/archived, it would just sit unread in `status.message`. The
        caller gets back the bare terminal event, with the summary text already archived as
        history's last entry.

        Called only after `execute` has already audited `result`, never before: `ActiveTask.
        _run_producer` awaits `execute` in full before it ever looks at the outcome, but
        `DefaultRequestHandler.on_message_send` doesn't wait for that - it returns as soon as its
        own subscription sees the terminal Task snapshot enqueued here, which reaches it through
        the event queue/consumer concurrently with `execute` still running. Auditing after
        enqueuing that terminal event would race the caller-visible response against `execute`'s
        own remaining code; a test driving the request through a short-lived event loop (e.g.
        starlette's TestClient without `with client:`, which tears its portal down as soon as the
        ASGI call returns) can then close out from under that task before the write finishes,
        dropping the record while the response the caller got was already correct. Auditing first
        makes the record exist before a caller can possibly observe "done".
        """
        is_success = result.outcome == Outcome.SUCCESS
        terminal_state = TaskState.TASK_STATE_COMPLETED if is_success else TaskState.TASK_STATE_FAILED
        terminal_text = _EXTRACTION_SUCCEEDED_MESSAGE if is_success else _EXTRACTION_FAILED_MESSAGE
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING, message=new_text_message(terminal_text)),
            ),
        )
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=terminal_state),
            ),
        )
        logger.info(
            "Intent extracted for %s: %s",
            self._agent.settings.intent_agent_name,
            result.text,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle task cancellation by emitting a CANCELED status event."""
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_CANCELED),
            ),
        )
