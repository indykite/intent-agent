"""A2A executor and Gemini-backed intent extraction pipeline.

Ported from the ``iag-demo/intent_agent`` prototype (see the Intent Agent Technical
Design Document). This is a structural port for this repository's package layout and
config model, not a rewrite of the extraction pipeline itself.
"""

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from a2a.helpers import new_task_from_user_message, new_text_artifact, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils.constants import DEFAULT_RPC_URL
from dogpile.cache.region import CacheRegion, make_region
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED

from indykite.intent_agent import auth, skills
from indykite.intent_agent.audit.message import (
    AuditMessage,
    Outcome,
    request_id_from_headers,
    schema_version,
    utc_timestamp,
)
from indykite.intent_agent.audit.sender import Sender, build_sender
from indykite.intent_agent.config import Settings
from indykite.intent_agent.errors import (
    IntentValidationError,
    SubjectTypeMismatchError,
    llm_timeout_error,
    no_intent_error,
    schema_not_populated_error,
    schema_unavailable_error,
    subject_type_mismatch_error,
    validation_feedback_error,
)
from indykite.intent_agent.input_schema import build_input_schema
from indykite.intent_agent.schema import SchemaClient
from indykite.intent_agent.subject_binding import (
    normalize_intent_response,
    schema_has_no_node_types,
    validate_subject_type,
)

logger = logging.getLogger(__name__)

_SECONDS_PER_MINUTE = 60
_MILLISECONDS_PER_SECOND = 1000
_NO_RESPONSE_TEXT = "(No response generated)"
_SKILL_ALREADY_ACTIVE = (
    "Skill 'extract-intent' is already active in this conversation - its instructions and the "
    "data schema are in the earlier tool result. Do not call activate_skill again; answer now "
    'with the {"evaluations"} JSON object.'
)


def _build_schema_cache(settings: Settings) -> CacheRegion:
    """Pick the schema cache backend named by CACHE_BACKEND: Redis, or in-process memory
    (the default) - single-instance deployments never need Redis running. Settings rejects
    CACHE_BACKEND=redis with no REDIS_URL at startup, so redis_url is guaranteed set here.

    Sets expiration_time from SCHEMA_REFRESH_TTL_MINUTES so SchemaClient.get_cached()
    refreshes on-demand once an entry ages past it, instead of a per-instance background
    timer - see get_cached()'s docstring for why that matters with multiple instances.
    """
    ttl_seconds = settings.schema_refresh_ttl_minutes * _SECONDS_PER_MINUTE
    logger.info("Using cache backend %r with TTL %d seconds", settings.cache_backend, ttl_seconds)
    if settings.cache_backend == "redis":
        return make_region().configure(
            "dogpile.cache.redis",
            expiration_time=ttl_seconds,
            # `url` (not host/port) so dogpile hands the DSN as-is to redis.StrictRedis.from_url,
            # preserving username/password/db/TLS scheme instead of dropping them.
            arguments={"url": str(settings.redis_url)},
        )
    return make_region().configure("dogpile.cache.memory", expiration_time=ttl_seconds)


class _ActivateSkillArgs(BaseModel):
    """Arguments schema for the activate_skill tool call."""

    name: str


def _dump(payload: dict[str, Any]) -> str:
    """Render a response payload the way every response text is rendered."""
    return json.dumps(payload, indent=2, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class RequestSubject:
    """The caller-supplied subject identity for one request (TDD 4.7) - resolved and sent by
    IAG itself (see IntentExecutor._subject_params), not derived from the Bearer token, which
    now authenticates the calling client rather than any subject. `id` may be blank (the
    Subject-Search exception); `type` may not.
    """

    id: str | None
    type: str


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """What one extraction produced: `text` is the A2A response body; the rest feeds the audit
    record (TDD 4.9) so the executor never has to parse its own output to classify it.
    `message` follows the IAG's rule for its `reason`: the same text the caller received.
    """

    text: str
    outcome: Outcome
    message: str = ""
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    validation_retries: int = 0
    schema_version: str = ""

    @classmethod
    def failed(
        cls,
        outcome: Outcome,
        error: dict[str, str],
        *,
        schema_version: str = "",
        validation_retries: int = 0,
    ) -> "ExtractionResult":
        """Wrap one of errors.py's 4.7 error shapes; its `reason` is the audit `message`."""
        return cls(
            text=_dump(error),
            outcome=outcome,
            message=error["reason"],
            schema_version=schema_version,
            validation_retries=validation_retries,
        )

    @classmethod
    def from_normalized(cls, text: str, *, schema_version: str, validation_retries: int) -> "ExtractionResult":
        """Classify normalize_intent_response()'s output: bound evaluations are a success, the
        4.7 no-intent shape (the LLM's own empty evaluations list) is `unmatched`.
        """
        data = json.loads(text)
        evaluations = data.get("evaluations")
        if isinstance(evaluations, list):
            # validate_evaluations already rejected anything that isn't an object, so the
            # list is exactly what the caller received.
            return cls(
                text=text,
                outcome=Outcome.SUCCESS,
                evaluations=evaluations,
                schema_version=schema_version,
                validation_retries=validation_retries,
            )
        return cls(
            text=text,
            outcome=Outcome.UNMATCHED,
            message=str(data.get("reason", "")),
            schema_version=schema_version,
            validation_retries=validation_retries,
        )

    @classmethod
    def free_text(cls, text: str, *, schema_version: str) -> "ExtractionResult":
        """A conversation that never activated extract-intent: the LLM's free text (or the
        no-response placeholder) is passed through and audited as `unmatched`, since no
        intent was extracted - not as an error, since nothing failed.
        """
        return cls(
            text=text or _NO_RESPONSE_TEXT,
            outcome=Outcome.UNMATCHED,
            message=text or _NO_RESPONSE_TEXT,
            schema_version=schema_version,
        )


class IntentAgent:
    """Owns the LLM, discovered skills, schema cache, and tool bindings for this run."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm: BaseChatModel | None = None,
        schema_client: SchemaClient | None = None,
    ) -> None:
        """Build the LLM, discover skills, and bind tools for this agent instance."""
        self.settings = settings
        self._llm = llm or ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            temperature=0,
        )
        self._schema_client = schema_client or SchemaClient(
            str(settings.indykite_base_url),
            settings.ik_app_agent_key.get_secret_value(),
            timeout_seconds=settings.schema_fetch_timeout_seconds,
            cache=_build_schema_cache(settings),
        )
        self.discovered_skills = skills.discover_skills()
        self.system_prompt = skills.build_skill_catalog_prompt(self.discovered_skills)
        self.skill_registry = {s["name"]: s for s in self.discovered_skills}
        self.agent_skills = skills.build_agent_skill_catalog(self.discovered_skills)

        self.tools = [
            t
            for t in (
                self._make_activate_skill_tool(),
                self._make_get_data_schema_tool(),
            )
            if t
        ]

    def preload_schema(self) -> None:
        """Fetch and cache the schema once, at startup - best effort, does not block startup.

        A missing schema at cold start (e.g. the platform is still coming up) no longer aborts
        the process: it keeps serving requests, and run_extraction() retries the fetch via
        get_cached_schema() on every request until the schema is populated. A schema that
        fetches successfully but has no node types beyond the platform's own bootstrap node
        (data model not populated - see schema_has_no_node_types) gets the same "keep
        retrying" treatment, logged distinctly from an actual fetch failure. subject_type is no
        longer static config - it arrives per-request from the caller (see
        IntentExecutor._subject_params) - so there is nothing to check against the schema at
        cold start any more; run_extraction validates it on every request instead.
        """
        self._schema_client.preload()
        schema = self.get_cached_schema()
        if schema is None:
            logger.warning(
                "Data schema unavailable at cold start - will keep retrying on every request until it's populated",
            )
            return
        if schema_has_no_node_types(schema):
            logger.warning(
                "Data schema fetched but the data model isn't populated yet at cold start - "
                "will keep retrying on every request until it's populated",
            )

    def get_cached_schema(self) -> dict[str, Any] | None:
        """Return the cached schema, fetching it once if not already cached."""
        return self._schema_client.get_cached()

    def _make_get_data_schema_tool(self) -> StructuredTool:
        """Build the get_data_schema tool bound to this agent's cached schema."""
        agent = self

        @tool
        def get_data_schema() -> str:
            """Return the cached graph schema from the Indykite platform as minified
            JSON Graph Format (JGF).

            Describes all Node types and Edge types with their properties. Node labels
            are entity types. Edge labels can be Actions or context, never Resources. Any
            node type can be a Resource. Nodes with system_labels containing DigitalTwin
            can also be Subjects.
            """
            schema = agent.get_cached_schema()
            return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)

        return get_data_schema

    def _make_activate_skill_tool(self) -> StructuredTool | None:
        """Create activate_skill; extract-intent injects the data schema."""
        if not self.skill_registry:
            return None
        valid_names = sorted(self.skill_registry.keys())
        description = (
            "Load the full instructions for a skill by name. Call this when a task "
            f"matches a skill's description. Valid names: {', '.join(valid_names)}."
        )

        def _invoke(name: str) -> str:
            """Return the requested skill's content, or an error if unknown."""
            name = name.strip()
            if name not in self.skill_registry:
                return f"Unknown skill: {name!r}. Valid skills: {', '.join(valid_names)}."
            logger.info("Activating skill: %s", name)
            skill = self.skill_registry[name]
            body = skill.get("body") or ""
            location = skill.get("location", "")
            skill_dir = str(Path(location).parent) if location else ""

            if name == "extract-intent":
                schema = self.get_cached_schema()
                preview_chars = self.settings.schema_preview_chars
                schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
                schema_block = (
                    "\n\n<data_schema>\n"
                    + schema_json[:preview_chars]
                    + "\n</data_schema>\n\nUse the above schema to map entity types."
                )
                return (
                    f"<skill_content name={name!r}>\n"
                    f"{schema_block}\n\n{body}\n\n"
                    f"Skill directory: {skill_dir}\n</skill_content>"
                )

            return f"<skill_content name={name!r}>\n{body}\n\nSkill directory: {skill_dir}\n</skill_content>"

        def _call(**kwargs: str) -> str:
            """Adapt the tool's keyword-argument call convention to _invoke."""
            return _invoke(kwargs["name"])

        return StructuredTool(
            name="activate_skill",
            description=description,
            args_schema=_ActivateSkillArgs,
            func=_call,
        )

    def _rpc_url(self) -> str:
        """Build the advertised JSON-RPC URL from the configured host and port."""
        return f"http://{self.settings.advertised_host}:{self.settings.intent_port}"

    def build_agent_card(self) -> AgentCard:
        """Build the A2A agent card advertising this agent's interface and skills."""
        return AgentCard(
            name=self.settings.intent_agent_name,
            description=(
                "Intent agent that extracts intent from a prompt and returns "
                "AuthZEN-compliant evaluation or search request objects. Uses the "
                "data schema to map entities to Subject, Action, and Resource. "
                "Requires a reachable data schema - requests fail if it's unavailable. "
                'On failure, returns {"prompt", "error", "reason", "obligation"} '
                'instead of "evaluations" - reason and obligation are natural-language '
                "strings meant to be fed to the caller's own LLM, e.g. to explain the "
                "failure to a user or ask them to clarify."
            ),
            version="1.0.0",
            provider={
                "organization": "Indykite",
                "url": "https://www.indykite.com",
            },
            capabilities=AgentCapabilities(
                streaming=False,
                push_notifications=False,
                extended_agent_card=False,
            ),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            supported_interfaces=[
                AgentInterface(
                    protocol_binding="JSONRPC",
                    url=self._rpc_url(),
                    protocol_version="1.0",
                ),
            ],
            skills=self.agent_skills,
        )

    def build_initial_messages(self, user_prompt: str) -> list[BaseMessage]:
        """Build the initial messages: skill catalog only (progressive disclosure)."""
        if not self.system_prompt:
            return [HumanMessage(content=user_prompt)]
        return [
            HumanMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

    @staticmethod
    def _response_text(response: BaseMessage) -> str:
        """Flatten the LLM's final response (plain text or content blocks) to one string."""
        raw_response_text = getattr(response, "text", "") or ""
        raw_content = getattr(response, "content", "") or ""
        if isinstance(raw_content, list) and raw_content and not raw_response_text:
            parts: list[str] = []
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    parts.append(getattr(block, "text", ""))
            return " ".join(parts) if parts else ""
        return str(raw_response_text or raw_content or "")

    def _dispatch_tool_call(
        self,
        call: dict[str, Any],
        messages: list[BaseMessage],
        *,
        extract_intent_active: bool = False,
    ) -> bool:
        """Invoke one tool call, append ToolMessage, report if extract-intent is now active.

        A repeat activate_skill for an already-active extract-intent gets a short "answer now"
        reminder instead of the full skill body plus schema again: re-injecting those every
        round bloats the context and is what an LLM stuck re-activating the skill until
        TOOL_CALL_MAX_ITERATIONS runs out looks like in practice.
        """
        name = call.get("name", "")
        args = call.get("args", {}) or {}
        call_id = call.get("id", "")
        activated_extract_intent = name == "activate_skill" and args.get("name") == "extract-intent"
        tool_obj = next((t for t in self.tools if t.name == name), None)
        if activated_extract_intent and extract_intent_active:
            logger.info("Skipping repeat activation of extract-intent - already active")
            result = _SKILL_ALREADY_ACTIVE
        elif tool_obj:
            try:
                result = tool_obj.invoke(args)
            except Exception as exc:  # noqa: BLE001 - tool errors are fed back to the LLM, not raised
                result = f"Error: {exc}"
        else:
            result = "Tool not found"
        messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
        return activated_extract_intent

    async def run_extraction(self, prompt: str, subject: RequestSubject) -> ExtractionResult:
        """Run the LLM tool-call loop for one prompt, return the final response and its audit
        classification (ExtractionResult).

        subject is supplied by the caller (IAG) per request - see
        IntentExecutor._subject_params - not read from static config or the Bearer token, which
        now authenticates the calling client rather than any subject.

        Checks schema availability and subject_type validity upfront, before any LLM call -
        neither can be verified without the schema, so there's no point spending an LLM call
        (or several, across tool-call iterations) on a result that would just be discarded
        afterward. A schema-unavailable request never populates the cache as a side effect of
        failing - get_cached_schema() retries the fetch itself on every call while the cache is
        empty, so the next request tries again on its own. A schema that fetches successfully
        but has no node types beyond the platform's own _Application node (data model not
        populated - see schema_has_no_node_types) is a distinct third state from both that and
        an actual subject_type mismatch: there's nothing yet to validate subject_type against,
        so it isn't a misconfiguration, but the schema itself isn't unreachable either. Both
        the unavailable and not-populated states self-heal without a restart, the same way any
        cached schema does, once a refresh (per-request retry, or the next request after
        SCHEMA_REFRESH_TTL_MINUTES elapses) picks up a real response.

        Binds tools/response_mime_type/response_schema here, per request, rather than once on
        self._llm at __init__ - build_input_schema(schema) constrains the LLM's final answer to
        the schema's actual node/edge vocabulary, but that schema can change between requests
        (TTL refresh, or cold-start recovery once the platform starts serving it), so binding it
        once at construction time would freeze the response shape to whatever schema existed at
        startup - possibly none at all. Rebuilding it fresh here (once per request, before the
        tool-call loop, not once per iteration) satisfies "derived ... each time it reloads"
        without a second cache to keep in sync.
        """
        schema = self.get_cached_schema()
        if schema is None:
            logger.error(
                "Schema unavailable - cannot verify subject_type %r; skipping extraction for prompt %r",
                subject.type,
                prompt,
            )
            return ExtractionResult.failed(Outcome.ERROR, schema_unavailable_error(prompt))
        version = schema_version(schema)
        if schema_has_no_node_types(schema):
            logger.error(
                "Schema fetched but the data model isn't populated yet - cannot verify "
                "subject_type %r; skipping extraction for prompt %r",
                subject.type,
                prompt,
            )
            return ExtractionResult.failed(Outcome.ERROR, schema_not_populated_error(prompt), schema_version=version)

        try:
            validate_subject_type(schema, subject.type)
        except SubjectTypeMismatchError as exc:
            logger.error(  # noqa: TRY400
                "subject_type %r invalid for the current schema; skipping extraction for prompt %r: %s",
                subject.type,
                prompt,
                exc,
            )
            return ExtractionResult.failed(Outcome.ERROR, subject_type_mismatch_error(prompt), schema_version=version)

        llm_with_tools = self._llm.bind_tools(
            self.tools,
            response_mime_type="application/json",
            response_schema=build_input_schema(schema),
        )
        try:
            async with asyncio.timeout(self.settings.request_timeout_seconds) as deadline:
                return await self._extract_with_retries(
                    llm_with_tools,
                    prompt,
                    subject=subject,
                    schema=schema,
                    version=version,
                )
        except TimeoutError:
            # Both the per-call and the whole-request limits raise TimeoutError; only the
            # request one marks its own context as expired.
            if deadline.expired():
                logger.error(  # noqa: TRY400 - a timeout's traceback carries no diagnostic value
                    "Request exceeded the %.1fs end-to-end budget; giving up on prompt %r",
                    self.settings.request_timeout_seconds,
                    prompt,
                )
            else:
                logger.error(  # noqa: TRY400 - a timeout's traceback carries no diagnostic value
                    "LLM call exceeded %.1fs; skipping extraction for prompt %r",
                    self.settings.llm_timeout_seconds,
                    prompt,
                )
            return ExtractionResult.failed(Outcome.ERROR, llm_timeout_error(prompt), schema_version=version)

    async def _invoke_llm(self, llm: Runnable[Any, BaseMessage], messages: list[BaseMessage]) -> BaseMessage:
        """One LLM call, bounded by LLM_TIMEOUT_SECONDS - raises TimeoutError past it."""
        async with asyncio.timeout(self.settings.llm_timeout_seconds):
            return await llm.ainvoke(messages)

    async def _run_tool_call_loop(
        self,
        llm: Runnable[Any, BaseMessage],
        messages: list[BaseMessage],
        *,
        extract_intent_active: bool = False,
    ) -> tuple[BaseMessage | None, bool]:
        """Invoke the LLM, dispatching tool calls, until it answers without any (or the
        TOOL_CALL_MAX_ITERATIONS budget runs out - then the response is None).

        Also reports whether extract-intent is active afterwards (it may already be, from an
        earlier attempt in the same conversation), since only then is the final answer an
        extraction to normalize rather than free text to pass through.
        """
        for _ in range(self.settings.tool_call_max_iterations):
            response = await self._invoke_llm(llm, messages)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                return response, extract_intent_active
            messages.append(response)
            for call in tool_calls:
                if self._dispatch_tool_call(call, messages, extract_intent_active=extract_intent_active):
                    extract_intent_active = True
        return None, extract_intent_active

    async def _extract_with_retries(
        self,
        llm: Runnable[Any, BaseMessage],
        prompt: str,
        *,
        subject: RequestSubject,
        schema: dict[str, Any],
        version: str,
    ) -> ExtractionResult:
        """Run the tool-call loop, validate the extraction, and retry with feedback (TDD 4.9).

        A validation failure (IntentValidationError: not JSON, or off-schema subject.type /
        action.name / resource.type) appends the rejected answer plus a corrective message
        (validation_feedback_error, in the fixed 4.7 error shape) to the same conversation
        and asks again, at most VALIDATION_MAX_RETRIES extra times - the retry budget is
        separate from TOOL_CALL_MAX_ITERATIONS, which bounds each attempt's tool-call rounds.
        The conversation (skill activation, injected schema) is kept across attempts so a
        retry doesn't re-spend tool calls re-activating the skill. Past the budget the caller
        gets no_intent_error, never the off-schema output. An extraction that exhausts the
        tool-call budget without ever answering gets no_intent_error too - the fixed 4.7
        shape, not the free-text placeholder reserved for a non-extraction conversation.
        """
        messages = self.build_initial_messages(prompt)
        activated_extract_intent = False
        max_retries = self.settings.validation_max_retries
        for attempt in range(1, max_retries + 2):
            retries = attempt - 1
            response, activated_extract_intent = await self._run_tool_call_loop(
                llm,
                messages,
                extract_intent_active=activated_extract_intent,
            )
            if response is None:
                if not activated_extract_intent:
                    return ExtractionResult.free_text("", schema_version=version)
                logger.error(
                    "Tool-call budget (%d) exhausted without an extraction answer; giving up on prompt %r",
                    self.settings.tool_call_max_iterations,
                    prompt,
                )
                return ExtractionResult.failed(
                    Outcome.ERROR,
                    no_intent_error(prompt),
                    schema_version=version,
                    validation_retries=retries,
                )
            final_text = self._response_text(response)
            if not activated_extract_intent:
                return ExtractionResult.free_text(final_text, schema_version=version)
            try:
                normalized = normalize_intent_response(
                    final_text,
                    prompt,
                    schema=schema,
                    subject_id=subject.id,
                    subject_type=subject.type,
                )
            except IntentValidationError as exc:
                logger.warning(
                    "Extraction attempt %d/%d failed validation for prompt %r: %s",
                    attempt,
                    max_retries + 1,
                    prompt,
                    exc,
                )
                feedback = _dump(validation_feedback_error(prompt, exc.errors))
                messages.extend((response, HumanMessage(content=feedback)))
            else:
                return ExtractionResult.from_normalized(normalized, schema_version=version, validation_retries=retries)
        logger.error("Extraction still off-schema after %d retries; giving up on prompt %r", max_retries, prompt)
        return ExtractionResult.failed(
            Outcome.VALIDATION_FAILED,
            no_intent_error(prompt),
            schema_version=version,
            validation_retries=max_retries,
        )


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
        """Validate the Bearer token identifying the calling client; 401 (HTTPException)
        otherwise. This token authenticates the caller (IAG), not any subject - it no longer
        supplies subject_id/subject_type, which now come from the caller-supplied message
        metadata instead (see _subject_params).
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
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    def _subject_params(self, context: RequestContext) -> tuple[str, str]:
        """Read subject_type/subject_id from the caller-supplied message metadata - IAG
        resolves the caller's own identity itself and supplies both per request, instead of
        subject_id coming from a token's `sub` claim and subject_type from static config.

        subject_type is required - 400 (HTTPException) if missing or blank (whitespace-only
        counts as blank), since every evaluation needs one. subject_id may be blank (the
        Subject-Search exception, TDD 4.7), so it isn't required here. Either present but not a
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
        """Run the extraction for an authenticated request, emitting the A2A task events."""
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
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
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


def build_app(
    settings: Settings,
    *,
    llm: BaseChatModel | None = None,
    schema_client: SchemaClient | None = None,
    audit_sender: Sender | None = None,
) -> tuple[Starlette, IntentAgent]:
    """Construct the Starlette app and its backing IntentAgent for these settings.

    The audit sender is built from AUDIT_DELIVERY here (or injected, for tests) and closed by
    the app's lifespan on shutdown - which is what drains a buffered (AUDIT_ASYNC) queue
    before the process exits, the same guarantee the IAG's sender gives on stop.
    """
    agent = IntentAgent(settings, llm=llm, schema_client=schema_client)
    agent.preload_schema()
    sender = audit_sender or build_sender(settings)

    agent_card = agent.build_agent_card()
    handler = DefaultRequestHandler(
        agent_executor=IntentExecutor(agent, sender),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    @contextlib.asynccontextmanager
    async def _lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await sender.close()

    app = Starlette(
        routes=[
            *create_agent_card_routes(agent_card=agent_card),
            *create_jsonrpc_routes(request_handler=handler, rpc_url=DEFAULT_RPC_URL),
        ],
        lifespan=_lifespan,
    )
    return app, agent
