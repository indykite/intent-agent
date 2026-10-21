"""Gemini-backed intent extraction pipeline: IntentAgent, its tools and tool-call loop, and the
ExtractionResult it hands to the A2A executor (executor.py).
"""

import asyncio
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from indykite.intent_agent import skills
from indykite.intent_agent.audit.message import Outcome, policy_version, schema_version
from indykite.intent_agent.config import Settings
from indykite.intent_agent.errors import (
    IntentValidationError,
    SubjectTypeMismatchError,
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
from indykite.intent_agent.policy import PolicyClient, build_policy_cache
from indykite.intent_agent.schema import SchemaClient, build_schema_cache
from indykite.intent_agent.subject_binding import (
    PolicyContext,
    RequestSubject,
    looks_like_evaluations,
    normalize_intent_response,
    rating,
    schema_has_no_node_types,
    validate_subject_type,
)

logger = logging.getLogger(__name__)


class _Phase(StrEnum):
    """Which run_extraction phase was underway when a TimeoutError fired; values are logged as-is."""

    SCHEMA_FETCH = "schema fetch"
    POLICY_FETCH = "policy fetch"
    LLM_INTERACTION = "LLM interaction"


_NO_RESPONSE_TEXT = "(No response generated)"
_EXTRACT_INTENT = "extract-intent"
_UNACTIVATED_EXTRACTION_NOTE = (
    "That answer was produced without activating the extract-intent skill, so its rules were not "
    "applied. The skill's instructions and the data schema follow; answer the original prompt "
    'again with the {"evaluations"} JSON object.\n\n'
)
_SKILL_ALREADY_ACTIVE = (
    "Skill 'extract-intent' is already active in this conversation - its instructions and the "
    "data schema are in the earlier tool result. Do not call activate_skill again; answer now "
    'with the {"evaluations"} JSON object.'
)


class _ActivateSkillArgs(BaseModel):
    """Arguments schema for the activate_skill tool call."""

    name: str


def _dump(payload: dict[str, Any]) -> str:
    """Render a response payload the way every response text is rendered."""
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _policy_block(policies: list[dict[str, Any]] | None, preview_chars: int) -> str:
    """Render the <policies> block for extract-intent's activation, whole policies only, up to
    `preview_chars`; empty without policies.
    """
    if not policies:
        return ""
    lines: list[str] = []
    for entry in policies:
        line = json.dumps(entry, separators=(",", ":"), ensure_ascii=False)
        if lines and sum(map(len, lines)) + len(line) > preview_chars:
            break
        lines.append(line)
    omitted = len(policies) - len(lines)
    more = f"\n({omitted} more policies not shown - call get_policies for all of them.)" if omitted else ""
    return (
        "\n\n<policies>\n"
        + "\n".join(lines)
        + more
        + "\n</policies>\n\nEvery evaluation must use one subject -ACTION-> resource combination these "
        "policies grant. Read each policy's condition.cypher to choose between them: it shows how the "
        "policy is applied to the graph, and so what a prompt asking for it is about."
    )


@dataclass(frozen=True, slots=True)
class _RequestContext:
    """This request's tool set, policy-fetch outcome and schema snapshot, built once per request."""

    tools: list[StructuredTool]
    policy_context: PolicyContext
    # The snapshot the tools were built from; _activate_and_ask_again re-injects this one.
    schema: dict[str, Any] = field(default_factory=dict)


def _policy_audit_fields(policy_context: PolicyContext | None) -> tuple[str, bool | None]:
    """Return (policy_version, policy_fetch_failed) for an ExtractionResult.

    Without a PolicyContext (no fetch attempted) this is ("", None): None, not False, which means
    attempted and succeeded.
    """
    if policy_context is None:
        return "", None
    return policy_version(policy_context.policies), policy_context.fetch_failed


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """One extraction's outcome: `text` is the A2A response body, the rest feeds the audit record.

    `message` is the same text the caller received; `confidence`/`explanation` are the model's
    rating, logged at DEBUG only.
    """

    text: str
    outcome: Outcome
    message: str = ""
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    validation_retries: int = 0
    schema_version: str = ""
    policy_version: str = ""
    # None = no policy fetch attempted, distinct from False (attempted and succeeded).
    policy_fetch_failed: bool | None = None
    confidence: str = ""
    explanation: str = ""

    @classmethod
    def failed(
        cls,
        outcome: Outcome,
        error: dict[str, str],
        *,
        schema_version: str = "",
        validation_retries: int = 0,
        policy_context: PolicyContext | None = None,
    ) -> "ExtractionResult":
        """Wrap one of errors.py's error shapes; its `reason` is the audit `message`."""
        policy_ver, policy_failed = _policy_audit_fields(policy_context)
        return cls(
            text=_dump(error),
            outcome=outcome,
            message=error["reason"],
            schema_version=schema_version,
            validation_retries=validation_retries,
            policy_version=policy_ver,
            policy_fetch_failed=policy_failed,
        )

    @classmethod
    def from_normalized(
        cls,
        text: str,
        llm_text: str,
        *,
        schema_version: str,
        validation_retries: int,
        policy_context: PolicyContext,
    ) -> "ExtractionResult":
        """Classify normalize_intent_response()'s output: evaluations are a success, the no-intent
        shape is `unmatched`; both carry the model's rating read from `llm_text`.
        """
        policy_ver, policy_failed = _policy_audit_fields(policy_context)
        data = json.loads(text)
        evaluations = data.get("evaluations")
        if isinstance(evaluations, list):
            confidence, explanation = rating(llm_text)
            return cls(
                text=text,
                outcome=Outcome.SUCCESS,
                evaluations=evaluations,
                schema_version=schema_version,
                validation_retries=validation_retries,
                policy_version=policy_ver,
                policy_fetch_failed=policy_failed,
                confidence=confidence,
                explanation=explanation,
            )
        return cls(
            text=text,
            outcome=Outcome.UNMATCHED,
            message=str(data.get("reason", "")),
            schema_version=schema_version,
            validation_retries=validation_retries,
            policy_version=policy_ver,
            policy_fetch_failed=policy_failed,
            explanation=rating(llm_text)[1],
        )

    @classmethod
    def free_text(
        cls,
        text: str,
        *,
        schema_version: str,
        policy_context: PolicyContext | None = None,
    ) -> "ExtractionResult":
        """Pass through a non-extraction conversation's free text, audited as `unmatched`."""
        policy_ver, policy_failed = _policy_audit_fields(policy_context)
        return cls(
            text=text or _NO_RESPONSE_TEXT,
            outcome=Outcome.UNMATCHED,
            message=text or _NO_RESPONSE_TEXT,
            schema_version=schema_version,
            policy_version=policy_ver,
            policy_fetch_failed=policy_failed,
        )


class IntentAgent:
    """Owns the LLM, discovered skills, schema cache, and tool bindings for this run."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm: BaseChatModel | None = None,
        schema_client: SchemaClient | None = None,
        policy_client: PolicyClient | None = None,
    ) -> None:
        """Build the LLM, discover skills, and set up the schema/policy clients.

        Tools are built per request, since policies depend on the caller's subject_type. The fetch
        executor is per instance because ThreadPoolExecutor.shutdown() is irreversible.
        """
        self.settings = settings
        self._fetch_executor = ThreadPoolExecutor(
            max_workers=settings.fetch_executor_max_workers, thread_name_prefix="intent-agent-fetch"
        )
        self._llm = llm or ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            temperature=0,
        )
        self._schema_client = schema_client or SchemaClient(
            str(settings.indykite_base_url),
            settings.ik_app_agent_key.get_secret_value(),
            timeout_seconds=settings.schema_fetch_timeout_seconds,
            cache=build_schema_cache(settings, self._fetch_executor),
        )
        self._policy_client = policy_client or PolicyClient(
            str(settings.indykite_base_url),
            settings.ik_app_agent_key.get_secret_value(),
            timeout_seconds=settings.policy_fetch_timeout_seconds,
            retry_attempts=settings.policy_fetch_retry_attempts,
            cache=build_policy_cache(settings),
        )
        self.discovered_skills = skills.discover_skills()
        self.system_prompt = skills.build_skill_catalog_prompt(self.discovered_skills)
        self.skill_registry = {s["name"]: s for s in self.discovered_skills}
        self.agent_skills = skills.build_agent_skill_catalog(self.discovered_skills)

    async def _run_in_fetch_executor[T](self, func: Callable[..., T], *args: object) -> T:
        """Run a blocking schema/policy fetch on the dedicated fetch executor.

        A hung fetch can't be preempted and keeps its worker busy past the request timeout; a
        separate pool keeps that pile-up from starving tool calls and audit writes.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._fetch_executor, func, *args)

    def preload_schema(self) -> None:
        """Fetch and cache the schema once at startup; best effort, never blocks startup.

        A missing or unpopulated schema is only logged; run_extraction retries it per request.
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
        """Return the cached schema, fetching it if not cached; None if unavailable.

        Call via _run_in_fetch_executor: even a cache hit can be Redis network I/O.
        """
        return self._schema_client.get_cached()

    def get_cached_policies(self, subject_type: str) -> tuple[list[dict[str, Any]] | None, bool]:
        """Return (policies, fetch_failed) for this subject_type, fetching if not cached.

        Check fetch_failed, not `policies is None`: a stale list may be served after a failed fetch.
        Call via _run_in_fetch_executor, like get_cached_schema.
        """
        return self._policy_client.get_cached(subject_type)

    def _make_get_data_schema_tool(self, schema: dict[str, Any]) -> StructuredTool:
        """Build the get_data_schema tool over this request's schema snapshot.

        Never re-fetches, so the tool shows the same schema the enum, validation and audit use.
        """
        schema_json = json.dumps(schema, separators=(",", ":"), ensure_ascii=False)

        @tool
        def get_data_schema() -> str:
            """Return the full cached graph schema from the Indykite platform as minified
            JSON Graph Format (JGF), with per-property value types and counts.

            Not needed for intent extraction: activating the extract-intent skill already
            provides every node type, property name and edge. Only call this afterwards, if a
            property's value type matters for a filter. Node labels are entity types. Edge
            labels show how the data is linked - never action names, which come only from the
            access policies.
            """
            return schema_json

        return get_data_schema

    def _make_get_policies_tool(
        self, policies: list[dict[str, Any]] | None, *, policy_fetch_failed: bool
    ) -> StructuredTool:
        """Build the get_policies tool over this request's policies snapshot.

        Never re-fetches, so the tool matches the enum, validation and audit. After a failed fetch it
        still returns the stale list (or []) and logs a warning when invoked.
        """
        policies_json = json.dumps(policies or [], separators=(",", ":"), ensure_ascii=False)

        @tool
        def get_policies() -> str:
            """Return this request's AuthZEN policies as minified JSON: a list of
            {"policy": <policy definition>, "tags": [...]}. Every evaluation must use one
            policy's subject.type, one of its "actions" and its resource.type together; its
            condition.cypher shows how the policy is applied to the graph.
            """
            if policy_fetch_failed:
                logger.warning("get_policies invoked but this request's policy fetch had failed")
            return policies_json

        return get_policies

    def _build_tools(
        self, schema: dict[str, Any], policies: list[dict[str, Any]] | None, *, policy_fetch_failed: bool
    ) -> list[StructuredTool]:
        """Build this request's tool set over the same schema/policies snapshot."""
        return [
            t
            for t in (
                self._make_activate_skill_tool(schema, policies),
                self._make_get_data_schema_tool(schema),
                self._make_get_policies_tool(policies, policy_fetch_failed=policy_fetch_failed),
            )
            if t
        ]

    def _skill_content(self, name: str, schema: dict[str, Any], policies: list[dict[str, Any]] | None) -> str:
        """Return the skill's activation content, or an error line for an unknown name.

        extract-intent gets the compact schema summary (capped at schema_preview_chars) and this
        request's policies block injected.
        """
        name = name.strip()
        if name not in self.skill_registry:
            return f"Unknown skill: {name!r}. Valid skills: {', '.join(sorted(self.skill_registry))}."
        logger.info("Activating skill: %s", name)
        skill = self.skill_registry[name]
        body = skill.get("body") or ""
        location = skill.get("location", "")
        skill_dir = str(Path(location).parent) if location else ""

        if name == _EXTRACT_INTENT:
            summary = render_schema_summary(schema, policies)[: self.settings.schema_preview_chars]
            schema_block = (
                "\n\n<data_schema>\n"
                + summary
                + "\n</data_schema>\n\nUse the above schema to map entity types and actions."
            )
            policy_block = _policy_block(policies, self.settings.policy_preview_chars)
            body = f"{schema_block}{policy_block}\n\n{body}"

        return f"<skill_content name={name!r}>\n{body}\n\nSkill directory: {skill_dir}\n</skill_content>"

    def _make_activate_skill_tool(
        self, schema: dict[str, Any], policies: list[dict[str, Any]] | None
    ) -> StructuredTool | None:
        """Create activate_skill over this request's schema/policies snapshot; None without skills."""
        if not self.skill_registry:
            return None
        valid_names = sorted(self.skill_registry.keys())
        description = (
            "Load the full instructions for a skill by name. Call this when a task "
            f"matches a skill's description. Valid names: {', '.join(valid_names)}."
        )

        def _call(**kwargs: str) -> str:
            """Adapt the tool's keyword-argument call convention to _skill_content."""
            return self._skill_content(kwargs["name"], schema, policies)

        return StructuredTool(
            name="activate_skill",
            description=description,
            args_schema=_ActivateSkillArgs,
            func=_call,
        )

    def _rpc_url(self) -> str:
        """Return the configured advertised JSON-RPC URL, else the plain-HTTP one on the listening port."""
        if self.settings.advertised_url is None:
            return f"http://intent:{self.settings.intent_port}"
        return str(self.settings.advertised_url)

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

    async def _dispatch_tool_call(
        self,
        call: dict[str, Any],
        messages: list[BaseMessage],
        tools: list[StructuredTool],
        *,
        extract_intent_active: bool = False,
    ) -> bool:
        """Invoke one tool call, append its ToolMessage, and report whether it activated extract-intent.

        A repeat activation gets a short "answer now" reminder instead of the skill body again, so
        the model can't loop re-activating until the tool budget runs out.
        """
        name = call.get("name", "")
        args = call.get("args", {}) or {}
        call_id = call.get("id", "")
        activated_extract_intent = name == "activate_skill" and args.get("name") == "extract-intent"
        tool_obj = next((t for t in tools if t.name == name), None)
        if activated_extract_intent and extract_intent_active:
            logger.info("Skipping repeat activation of extract-intent - already active")
            result = _SKILL_ALREADY_ACTIVE
        elif tool_obj:
            try:
                result = await tool_obj.ainvoke(args)
            except Exception as exc:  # noqa: BLE001 - tool errors are fed back to the LLM, not raised
                result = f"Error: {exc}"
        else:
            result = "Tool not found"
        messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
        return activated_extract_intent

    async def run_extraction(self, prompt: str, subject: RequestSubject) -> ExtractionResult:
        """Run the extraction for one prompt and return the response with its audit classification.

        Schema, subject_type and policies are checked before any LLM call; tools and the response
        schema are bound per request because the schema can change. The whole run is bounded by
        REQUEST_TIMEOUT_SECONDS.
        """
        version = ""
        policy_context = PolicyContext(policies=None, fetch_failed=True)
        phase = _Phase.SCHEMA_FETCH
        try:
            async with asyncio.timeout(self.settings.request_timeout_seconds) as deadline:
                schema = await self._run_in_fetch_executor(self.get_cached_schema)
                refusal = self._schema_refusal(schema, subject.type, prompt)
                if refusal is not None:
                    return refusal
                version = schema_version(schema)

                phase = _Phase.POLICY_FETCH
                policies, policy_fetch_failed = await self._run_in_fetch_executor(
                    self.get_cached_policies, subject.type
                )
                policy_context = PolicyContext(policies=policies, fetch_failed=policy_fetch_failed)
                error = self._policy_refusal(policy_context, subject.type, prompt)
                if error is not None:
                    return ExtractionResult.failed(
                        Outcome.ERROR, error, schema_version=version, policy_context=policy_context
                    )
                llm_with_tools, request = self._bind_request(schema, policy_context)
                phase = _Phase.LLM_INTERACTION
                return await self._extract_with_retries(llm_with_tools, prompt, subject=subject, request=request)
        except TimeoutError:
            return self._timeout_result(phase, prompt, deadline, version, policy_context)

    @staticmethod
    def _schema_refusal(schema: dict[str, Any] | None, subject_type: str, prompt: str) -> ExtractionResult | None:
        """Return the failed result when the schema is missing, empty or lacks the caller's type, else None."""
        if schema is None:
            logger.error(
                "Schema unavailable - cannot verify subject_type %r; skipping extraction for prompt %r",
                subject_type,
                prompt,
            )
            return ExtractionResult.failed(Outcome.ERROR, schema_unavailable_error(prompt))
        if schema_has_no_node_types(schema):
            logger.warning(
                "Schema fetched but the data model isn't populated yet - cannot verify "
                "subject_type %r; skipping extraction for prompt %r",
                subject_type,
                prompt,
            )
            return ExtractionResult.failed(
                Outcome.ERROR, schema_not_populated_error(prompt), schema_version=schema_version(schema)
            )
        try:
            validate_subject_type(schema, subject_type)
        except SubjectTypeMismatchError as exc:
            logger.error(  # noqa: TRY400
                "subject_type %r invalid for the current schema; skipping extraction for prompt %r: %s",
                subject_type,
                prompt,
                exc,
            )
            return ExtractionResult.failed(
                Outcome.ERROR, subject_type_mismatch_error(prompt), schema_version=schema_version(schema)
            )
        return None

    def _bind_request(
        self, schema: dict[str, Any], policy_context: PolicyContext
    ) -> tuple[Runnable[Any, BaseMessage], _RequestContext]:
        """Bind this request's tools and structured-output schema to the LLM."""
        policies = policy_context.policies
        tools = self._build_tools(schema, policies, policy_fetch_failed=policy_context.fetch_failed)
        llm_with_tools = self._llm.bind_tools(
            tools,
            response_mime_type="application/json",
            response_schema=build_input_schema(schema, policies),
        )
        return llm_with_tools, _RequestContext(tools=tools, policy_context=policy_context, schema=schema)

    @staticmethod
    def _policy_refusal(policy_context: PolicyContext, subject_type: str, prompt: str) -> dict[str, str] | None:
        """Return policy_unavailable_error when the caller's type has no policies, else None.

        Stale policies after a failed fetch proceed, with a warning.
        """
        if policy_context.policies is None:
            logger.warning(
                "Policy fetch failed for subject_type %r and no policies are cached; skipping extraction for prompt %r",
                subject_type,
                prompt,
            )
            return policy_unavailable_error(prompt)
        if not policy_context.policies:
            logger.warning(
                "No access policies exist for subject_type %r; skipping extraction for prompt %r", subject_type, prompt
            )
            return policy_unavailable_error(prompt)
        if policy_context.fetch_failed:
            logger.warning(
                "Policy fetch failed for subject_type %r; proceeding with last known-good policies for prompt %r",
                subject_type,
                prompt,
            )
        return None

    def _timeout_result(
        self,
        phase: _Phase,
        prompt: str,
        deadline: asyncio.Timeout,
        version: str,
        policy_context: PolicyContext,
    ) -> ExtractionResult:
        """Build run_extraction's response to a TimeoutError, classified by `phase`.

        Only the LLM phase reports llm_timeout_error; a slow schema or policy fetch gets
        request_timeout_error so the model isn't blamed.
        """
        if phase != _Phase.LLM_INTERACTION:
            logger.error(
                "%s exceeded the %.1fs end-to-end budget; giving up on prompt %r",
                phase.capitalize(),
                self.settings.request_timeout_seconds,
                prompt,
            )
            if phase == _Phase.SCHEMA_FETCH:
                return ExtractionResult.failed(Outcome.ERROR, request_timeout_error(prompt))
            return ExtractionResult.failed(
                Outcome.ERROR,
                request_timeout_error(prompt),
                schema_version=version,
                policy_context=policy_context,
            )
        # Both limits raise TimeoutError; only the whole-request one marks its deadline expired.
        if deadline.expired():
            logger.error(
                "Request exceeded the %.1fs end-to-end budget; giving up on prompt %r",
                self.settings.request_timeout_seconds,
                prompt,
            )
        else:
            logger.error(
                "LLM call exceeded %.1fs; skipping extraction for prompt %r",
                self.settings.llm_timeout_seconds,
                prompt,
            )
        return ExtractionResult.failed(
            Outcome.ERROR,
            llm_timeout_error(prompt),
            schema_version=version,
            policy_context=policy_context,
        )

    async def _invoke_llm(self, llm: Runnable[Any, BaseMessage], messages: list[BaseMessage]) -> BaseMessage:
        """One LLM call, bounded by LLM_TIMEOUT_SECONDS - raises TimeoutError past it."""
        async with asyncio.timeout(self.settings.llm_timeout_seconds):
            return await llm.ainvoke(messages)

    async def _run_tool_call_loop(
        self,
        llm: Runnable[Any, BaseMessage],
        messages: list[BaseMessage],
        tools: list[StructuredTool],
        *,
        extract_intent_active: bool = False,
    ) -> tuple[BaseMessage | None, bool]:
        """Invoke the LLM, dispatching tool calls, until it answers without any.

        Returns (response, extract_intent_active); response is None if TOOL_CALL_MAX_ITERATIONS
        runs out.
        """
        for _ in range(self.settings.tool_call_max_iterations):
            response = await self._invoke_llm(llm, messages)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                return response, extract_intent_active
            messages.append(response)
            for call in tool_calls:
                if await self._dispatch_tool_call(call, messages, tools, extract_intent_active=extract_intent_active):
                    extract_intent_active = True
        return None, extract_intent_active

    async def _activate_and_ask_again(
        self,
        llm: Runnable[Any, BaseMessage],
        messages: list[BaseMessage],
        response: BaseMessage,
        prompt: str,
        request: _RequestContext,
    ) -> tuple[BaseMessage | None, bool]:
        """Inject extract-intent after an {evaluations} answer given without activating it, and ask again.

        Such an answer is unvalidated and must never pass through as free text.
        """
        logger.info(
            "Answered without activating extract-intent; injecting the skill and asking again for prompt %r", prompt
        )
        content = _UNACTIVATED_EXTRACTION_NOTE + self._skill_content(
            _EXTRACT_INTENT, request.schema, request.policy_context.policies
        )
        messages.extend((response, HumanMessage(content=content)))
        return await self._run_tool_call_loop(llm, messages, request.tools, extract_intent_active=True)

    async def _extract_with_retries(
        self,
        llm: Runnable[Any, BaseMessage],
        prompt: str,
        *,
        subject: RequestSubject,
        request: _RequestContext,
    ) -> ExtractionResult:
        """Run the tool-call loop, validate the extraction, and retry with feedback.

        Retries up to VALIDATION_MAX_RETRIES in the same conversation; past that, or on an exhausted
        tool-call budget mid-extraction, returns no_intent_error, never off-schema output.
        """
        version = schema_version(request.schema)
        messages = self.build_initial_messages(prompt)
        activated_extract_intent = False
        max_retries = self.settings.validation_max_retries
        for attempt in range(1, max_retries + 2):
            retries = attempt - 1
            response, activated_extract_intent = await self._run_tool_call_loop(
                llm,
                messages,
                request.tools,
                extract_intent_active=activated_extract_intent,
            )
            if response is not None and not activated_extract_intent:
                final_text = self._response_text(response)
                if not looks_like_evaluations(final_text):
                    return ExtractionResult.free_text(
                        final_text,
                        schema_version=version,
                        policy_context=request.policy_context,
                    )
                response, activated_extract_intent = await self._activate_and_ask_again(
                    llm, messages, response, prompt, request
                )
            if response is None:
                return self._no_answer_result(
                    prompt, request, retries, activated_extract_intent=activated_extract_intent
                )
            final_text = self._response_text(response)
            try:
                return self._validated_result(final_text, prompt, subject=subject, request=request, retries=retries)
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
        logger.error("Extraction still off-schema after %d retries; giving up on prompt %r", max_retries, prompt)
        return ExtractionResult.failed(
            Outcome.VALIDATION_FAILED,
            no_intent_error(prompt),
            schema_version=version,
            validation_retries=max_retries,
            policy_context=request.policy_context,
        )

    def _no_answer_result(
        self, prompt: str, request: _RequestContext, retries: int, *, activated_extract_intent: bool
    ) -> ExtractionResult:
        """The result when the tool-call budget ran out: empty free text, or no_intent_error mid-extraction."""
        version = schema_version(request.schema)
        if not activated_extract_intent:
            return ExtractionResult.free_text("", schema_version=version, policy_context=request.policy_context)
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
            policy_context=request.policy_context,
        )

    @staticmethod
    def _validated_result(
        final_text: str, prompt: str, *, subject: RequestSubject, request: _RequestContext, retries: int
    ) -> ExtractionResult:
        """Normalize the extraction answer into a result; raises IntentValidationError when it is off-schema."""
        normalized = normalize_intent_response(
            final_text,
            prompt,
            schema=request.schema,
            subject=subject,
            policy_context=request.policy_context,
        )
        result = ExtractionResult.from_normalized(
            normalized,
            final_text,
            schema_version=schema_version(request.schema),
            validation_retries=retries,
            policy_context=request.policy_context,
        )
        if result.explanation:
            verdict = f"confidence {result.confidence}" if result.confidence else "refused"
            logger.debug("Extraction %s for prompt %r: %s", verdict, prompt, result.explanation)
        return result
