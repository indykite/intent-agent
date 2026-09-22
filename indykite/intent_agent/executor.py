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
            llm_provider=_llm_provider(self._settings.gemini_model),
            llm_model=self._settings.gemini_model,
            latency_ms=latency_ms,
            validation_retries=result.validation_retries,
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
        """
        audit_record = _RequestAudit(self._agent.settings, context)
        try:
            await self._authenticate(context)
            audit_record.subject_type, audit_record.subject_id = self._subject_params(context)
            result = await self._extract(
                context,
                event_queue,
                audit_record.prompt,
                subject_id=audit_record.subject_id,
                subject_type=audit_record.subject_type,
            )
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
                introspect_url=str(settings.introspect_url),
                client_id=settings.introspect_client_id,
                client_secret=settings.introspect_client_secret.get_secret_value(),
                timeout_seconds=settings.introspect_timeout_seconds,
            )
        except auth.TokenValidationError as exc:
            raise HTTPException(status_code=_TOKEN_FAILURE_STATUS[exc.failure], detail=str(exc)) from exc

    def _subject_params(self, context: RequestContext) -> tuple[str, str]:
        """Read subject_type/subject_id from the caller-supplied message metadata - IAG
        resolves the caller's own identity itself and supplies both per request, instead of
        subject_id coming from a token's `sub` claim and subject_type from static config.

        subject_type is required - 400 (HTTPException) if missing or blank (whitespace-only
        counts as blank), since every evaluation needs one. subject_id may be blank (TDD 4.7
        doesn't require the caller to have one - a Subject Search never binds it either way,
        see subject_binding._subject_search), so it isn't required here. Either present but not a
        string (e.g. a nested object) is also a 400 - str(some_dict) would otherwise silently
        coerce it into a Python repr and bind that into every evaluation and audit record,
        instead of surfacing the malformed request to the caller.
        """
        metadata = dict(context.message.metadata) if context.message and context.message.metadata else {}
        subject_type = self._require_metadata_string(metadata, "subject_type")
        subject_id = self._require_metadata_string(metadata, "subject_id")
        if not subject_type:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="subject_type is required")
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

    async def _extract(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        raw_text: str,
        *,
        subject_id: str,
        subject_type: str,
    ) -> ExtractionResult:
        """Run the extraction for an authenticated request, emitting the A2A task events.

        The terminal TaskState reflects the outcome (ENG-9804): TASK_STATE_COMPLETED only for
        a successful extraction, TASK_STATE_FAILED for every other outcome (validation failure,
        no-intent/unmatched, timeout, schema unavailable, ...) - the caller's artifact JSON
        still carries the fixed {prompt, error, reason, obligation} shape (errors.py) either way,
        this only changes what the task's own status reports. A fixed summary message (success
        or failure - the detailed reason stays in the artifact JSON's `reason` field, not
        duplicated here) ends up as the task's last `history` entry, replacing the pre-extraction
        "Extracting intent..." placeholder as the final word instead of leaving it there - done
        by emitting the terminal status twice (see the comment at the call site for why).
        """
        logger.info(
            "Received message for %s: %s",
            self._agent.settings.intent_agent_name,
            raw_text,
        )

        prompt = raw_text or "(empty)"

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

        result = await self._agent.run_extraction(prompt, RequestSubject(id=subject_id, type=subject_type))

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                artifact=new_text_artifact(name="result", text=result.text),
            ),
        )
        # ENG-9804: the task's own terminal state must tell success from failure - a caller
        # otherwise has no way to know without parsing the artifact JSON for an "error" key.
        # VALIDATION_FAILED/UNMATCHED/ERROR (no_intent_error, timeouts, schema unavailable, ...)
        # all surface as TASK_STATE_FAILED; only SUCCESS is TASK_STATE_COMPLETED. A fixed summary
        # message (the detailed reason stays in the artifact JSON's `reason` field, not
        # duplicated here) must land in the task's `history`, not just sit on the terminal status -
        # a2a's TaskManager only archives a status's message into history once a *later*
        # TaskStatusUpdateEvent supersedes it, so the message is carried on one last WORKING
        # update first (never on the terminal one itself: DefaultRequestHandler.on_message_send
        # returns on the *first* terminal-state Task snapshot it sees, so a message on that event
        # would never get superseded/archived - it would just sit unread in `status.message`).
        # The final, bare terminal event is what the caller actually gets back, with the summary
        # text already archived as history's last entry instead of "Extracting intent...".
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
        return result

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle task cancellation by emitting a CANCELED status event."""
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_CANCELED),
            ),
        )
