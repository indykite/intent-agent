"""A2A executor: authenticates the caller, runs the extraction, emits the task events, and
audits every request exactly once (TDD 4.9).
"""

import logging
import time
from typing import Any

from a2a.helpers import new_task_from_user_message, new_text_artifact, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskArtifactUpdateEvent, TaskState, TaskStatus, TaskStatusUpdateEvent
from starlette.exceptions import HTTPException
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_502_BAD_GATEWAY,
)

from indykite.intent_agent import auth
from indykite.intent_agent.agent import ExtractionResult, IntentAgent, RequestSubject
from indykite.intent_agent.audit.message import AuditMessage, Outcome, request_id_from_headers, utc_timestamp
from indykite.intent_agent.audit.sender import Sender
from indykite.intent_agent.config import Settings

logger = logging.getLogger(__name__)

_MILLISECONDS_PER_SECOND = 1000
_EXTRACTION_SUCCEEDED_MESSAGE = "Intent extracted successfully."
_EXTRACTION_FAILED_MESSAGE = "Failed to extract intent."

# What each class of token failure (auth.TokenFailure) is answered with: 401 only
# for what the caller got wrong, 500 for this service's own failure, 502 for an IdP that answered
# something other than an RFC 7662 answer. A 401 for an IdP outage would send the caller off to
# refresh a token that was never the problem.
_TOKEN_FAILURE_STATUS = {
    auth.TokenFailure.CALLER: HTTP_401_UNAUTHORIZED,
    auth.TokenFailure.SERVICE: HTTP_500_INTERNAL_SERVER_ERROR,
    auth.TokenFailure.PROVIDER: HTTP_502_BAD_GATEWAY,
}


def _prompt_from_context(context: RequestContext) -> str:
    """Concatenate the text parts of the incoming A2A message."""
    raw_text = ""
    if context.message:
        for part in context.message.parts or []:
            text = getattr(part, "text", None)
            if text:
                raw_text += text
    return raw_text


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

    def __init__(self, settings: Settings, context: RequestContext) -> None:
        """Capture what's known before authentication: ids, prompt, and the start time."""
        self._settings = settings
        self._started = time.perf_counter()
        headers = context.call_context.state.get("headers") if context.call_context else None
        self.request_id = request_id_from_headers(headers or {})
        self.context_id = context.context_id or ""
        self.prompt = _prompt_from_context(context)
        self.subject_id = ""
        self.subject_type = ""

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

        A rejected token is audited too, like the IAG audits every rejection - as
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
        audit_record = _RequestAudit(self._agent.settings, context)
        try:
            await self._authenticate(context)
            audit_record.subject_type, audit_record.subject_id = self._subject_params(context)
            result = await self._run_extraction(context, event_queue, audit_record)
        except HTTPException as exc:
            await self._audit(audit_record, ExtractionResult(text="", outcome=Outcome.ERROR, message=str(exc.detail)))
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
        itself produced: the 401 / exception being re-raised in `execute`, or a result whose
        artifact the caller already has. An audit failure is logged with the request id instead.
        """
        try:
            await self._audit_sender.send(audit_record.record(result))
        except Exception:
            logger.exception("Failed to audit request %s (outcome %s)", audit_record.request_id, result.outcome)

    async def _authenticate(self, context: RequestContext) -> None:
        """Validate the Bearer token identifying the calling client; HTTPException otherwise -
        401 when the token itself is missing or inactive, 500 when this service couldn't get an
        answer about it, 502 when the IdP's answer can't be used (_TOKEN_FAILURE_STATUS). This
        token authenticates the caller (IAG), not any subject - it no longer supplies
        subject_id/subject_type, which now come from the caller-supplied message metadata
        instead (see _subject_params).
        """
        access_token = auth.get_access_token_from_context(context)
        if not access_token:
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Authorization required")

        settings = self._agent.settings
        try:
            await auth.validate_access_token(
                access_token,
                auth.IntrospectionConfig(
                    introspect_url=str(settings.introspect_url),
                    client_id=settings.introspect_client_id,
                    client_secret=settings.introspect_client_secret.get_secret_value(),
                    expected_audience=settings.expected_token_audience,
                    timeout_seconds=settings.introspect_timeout_seconds,
                ),
            )
        except auth.TokenValidationError as exc:
            raise HTTPException(status_code=_TOKEN_FAILURE_STATUS[exc.failure], detail=str(exc)) from exc

    def _subject_params(self, context: RequestContext) -> tuple[str, str]:
        """Read subject_type/subject_id from the caller-supplied message metadata - IAG
        resolves the caller's own identity itself and supplies both per request, instead of
        subject_id coming from a token's `sub` claim and subject_type from static config.

        Both are required (ENG-9810) - 400 (HTTPException) if either is missing or blank
        (whitespace-only counts as blank). subject_id used to be reliably populated from the
        token's own `sub` claim, so a real authenticated caller was never actually anonymous in
        practice; now that IAG must supply it explicitly instead, a missing subject_id is a
        caller-side bug (it forgot to resolve/attach the caller's identity), not a legitimate
        anonymous-caller request, and must be rejected the same as a missing subject_type rather
        than silently proceeding with an unidentified caller. Either present but not a string
        (e.g. a nested object) is also a 400 - str(some_dict) would otherwise silently coerce it
        into a Python repr and bind that into every evaluation and audit record, instead of
        surfacing the malformed request to the caller.
        """
        metadata = dict(context.message.metadata) if context.message and context.message.metadata else {}
        subject_type = self._require_metadata_string(metadata, "subject_type")
        subject_id = self._require_metadata_string(metadata, "subject_id")
        if not subject_type:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="subject_type is required")
        if not subject_id:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="subject_id is required")
        return subject_type, subject_id

    @staticmethod
    def _require_metadata_string(metadata: dict[str, Any], key: str) -> str:
        """Return metadata[key] as a stripped string, or "" if absent/blank - raising 400 if
        it's present but not a string at all (see _subject_params).
        """
        value = metadata.get(key)
        if value is None:
            return ""
        if not isinstance(value, str):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"{key} must be a string")
        return value.strip()

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
