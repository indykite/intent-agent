"""Gemini-backed intent extraction pipeline: the IntentAgent, its tools and tool-call loop, and
the ExtractionResult it hands to the A2A executor (executor.py). build_app (app.py) wires it up.

Ported from the ``iag-demo/intent_agent`` prototype (see the Intent Agent Technical
Design Document). This is a structural port for this repository's package layout and
config model, not a rewrite of the extraction pipeline itself.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
    request_timeout_error,
    schema_not_populated_error,
    schema_unavailable_error,
    subject_type_mismatch_error,
    validation_feedback_error,
)
from indykite.intent_agent.input_schema import build_input_schema
from indykite.intent_agent.policy import PolicyClient, build_policy_cache
from indykite.intent_agent.schema import SchemaClient, build_schema_cache
from indykite.intent_agent.subject_binding import (
    PolicyContext,
    RequestSubject,
    normalize_intent_response,
    schema_has_no_node_types,
    validate_subject_type,
)

logger = logging.getLogger(__name__)

_FETCH_EXECUTOR_MAX_WORKERS = 8


_NO_RESPONSE_TEXT = "(No response generated)"
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
    """Build the <policies> context block for extract-intent's activation (ENG-9671),
    mirroring the <data_schema> block. Empty string when there are no policies - the fetch
    failed, or none exist for this subject_type - so the LLM falls back to schema-only action
    derivation exactly as it did before this feature existed.
    """
    if not policies:
        return ""
    policies_json = json.dumps(policies, indent=2, ensure_ascii=False)
    return (
        "\n\n<policies>\n"
        + policies_json[:preview_chars]
        + "\n</policies>\n\nUse the above policies, when relevant, to determine action.name - "
        'mark such evaluations with context.parameters.action_derivation_rule set to "policy_based".'
    )


@dataclass(frozen=True, slots=True)
class _RequestContext:
    """This request's tool set and policy-fetch outcome, bundled - both come from the one
    per-request PolicyClient.get_cached() call in run_extraction (policies vary by the
    caller's subject_type, unlike the process-wide schema, so tools/policy_context can't be
    built once at __init__ like they used to be), and _extract_with_retries/
    _run_tool_call_loop/_dispatch_tool_call would otherwise each take on two more arguments,
    pushing them over the lint limit.
    """

    tools: list[StructuredTool]
    policy_context: PolicyContext


def _policy_audit_fields(policy_context: PolicyContext | None) -> tuple[str, bool]:
    """(policy_version, policy_fetch_failed) for an ExtractionResult, from this request's
    PolicyContext - None when the result was built before any policy fetch was attempted (the
    fail-fast paths in run_extraction, before get_cached_policies is even called), same as
    schema_version defaulting to "" for those same paths.
    """
    if policy_context is None:
        return "", False
    return policy_version(policy_context.policies), policy_context.fetch_failed


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
    policy_version: str = ""
    policy_fetch_failed: bool = False

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
        """Wrap one of errors.py's 4.7 error shapes; its `reason` is the audit `message`."""
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
        *,
        schema_version: str,
        validation_retries: int,
        policy_context: PolicyContext,
    ) -> "ExtractionResult":
        """Classify normalize_intent_response()'s output: bound evaluations are a success, the
        4.7 no-intent shape (the LLM's own empty evaluations list) is `unmatched`.
        """
        policy_ver, policy_failed = _policy_audit_fields(policy_context)
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
                policy_version=policy_ver,
                policy_fetch_failed=policy_failed,
            )
        return cls(
            text=text,
            outcome=Outcome.UNMATCHED,
            message=str(data.get("reason", "")),
            schema_version=schema_version,
            validation_retries=validation_retries,
            policy_version=policy_ver,
            policy_fetch_failed=policy_failed,
        )

    @classmethod
    def free_text(
        cls,
        text: str,
        *,
        schema_version: str,
        policy_context: PolicyContext | None = None,
    ) -> "ExtractionResult":
        """A conversation that never activated extract-intent: the LLM's free text (or the
        no-response placeholder) is passed through and audited as `unmatched`, since no
        intent was extracted - not as an error, since nothing failed.
        """
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
        """Build the LLM, discover skills, and set up the schema/policy clients for this
        agent instance. Tools are built per-request (see _build_tools), not here - unlike the
        schema, policies vary by the caller's subject_type, so there is nothing to bind them
        to until a request arrives.

        `_fetch_executor` is owned by this instance, not a module-level singleton (review) - a
        shared one would tie every IntentAgent's schema/policy fetches to whichever instance's
        app.py lifespan happened to shut down first: ThreadPoolExecutor.shutdown() is not
        reversible, so a second app or embedded caller sharing that pool would fail with
        "cannot schedule new futures after shutdown" the moment any one of them stopped.
        """
        self.settings = settings
        self._fetch_executor = ThreadPoolExecutor(
            max_workers=_FETCH_EXECUTOR_MAX_WORKERS, thread_name_prefix="intent-agent-fetch"
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
        """Run a blocking schema/policy fetch call on this instance's dedicated
        `_fetch_executor`, not the event loop's shared default one that every tool call and the
        audit sender's file write also use (review, ENG-9671). Python cannot preempt an
        already-running synchronous call, so a stuck/unreachable schema or policy endpoint
        keeps its worker running for its own timeout x retry course even after asyncio.timeout
        has given up on the awaiting request - see run_extraction's docstring. Enough
        concurrent requests against a genuinely down endpoint could pile up orphaned workers
        and exhaust a *shared* executor, starving unrelated concurrent requests' tool calls and
        audit writes too; a small, dedicated pool for fetches specifically (which already
        degrade gracefully - schema_unavailable_error, or policy_fetch_failed=True proceeding
        schema-only) contains that pile-up instead of spreading it to everything else a request
        depends on.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._fetch_executor, func, *args)

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
        """Return the cached schema, fetching it once if not already cached.

        Always dispatched through `_run_in_fetch_executor`, never called directly on the event
        loop (review) - even a cache hit is a synchronous `self._cache.get(...)` call, and with
        CACHE_BACKEND=redis that is real network I/O, not a bounded, in-process lookup. A slow
        or unreachable Redis would otherwise block the loop for every concurrent request this
        process is serving, and prevent REQUEST_TIMEOUT_SECONDS's own asyncio.timeout from ever
        getting a turn to fire - the same failure mode _run_in_fetch_executor already exists to
        avoid for the platform's own HTTP calls.
        """
        return self._schema_client.get_cached()

    def get_cached_policies(self, subject_type: str) -> list[dict[str, Any]] | None:
        """Return this subject_type's cached AuthZEN policies (ENG-9671), fetching them once if
        not already cached. None means the fetch failed after retries, not "no policies exist"
        (an empty list) - run_extraction tells the two apart (see PolicyContext.fetch_failed).

        Always dispatched through `_run_in_fetch_executor`, never called directly on the event
        loop - see get_cached_schema's docstring; the same Redis-backend risk applies here too.
        """
        return self._policy_client.get_cached(subject_type)

    def _make_get_data_schema_tool(self, schema: dict[str, Any]) -> StructuredTool:
        """Build the get_data_schema tool, returning this request's own schema snapshot
        (review) - the exact one run_extraction already used to build the response schema's
        enums - rather than an independent get_cached_schema() call. A fresh call could return
        whatever the TTL has since refreshed to: a schema the bound response schema's enum and
        validate_evaluations were never built against, so the tool could show the LLM a type it
        can neither emit (outside its own structured-output enum) nor have accepted
        (validate_evaluations still checks the original snapshot) - and leave the audit's
        schema_version hash out of step with what the tool actually told the model.
        """
        schema_json = json.dumps(schema, separators=(",", ":"), ensure_ascii=False)

        @tool
        def get_data_schema() -> str:
            """Return the cached graph schema from the Indykite platform as minified
            JSON Graph Format (JGF).

            Describes all Node types and Edge types with their properties. Node labels
            are entity types. Edge labels can be Actions or context, never Resources. Any
            node type can be a Resource. Nodes with system_labels containing DigitalTwin
            can also be Subjects.
            """
            return schema_json

        return get_data_schema

    def _make_get_policies_tool(self, policies: list[dict[str, Any]] | None) -> StructuredTool:
        """Build the get_policies tool (ENG-9671), returning this request's own policies
        snapshot - the exact one run_extraction already used to build request.policy_context
        and the response schema's action enum - rather than an independent
        get_cached_policies() call (review). Same rationale as _make_get_data_schema_tool: a
        fresh call could return a cache entry refreshed or evicted since that snapshot was
        taken, showing the LLM a policy action neither in its own enum nor accepted by
        validate_evaluations, with the audit's policy_version hash left describing a
        different, stale policy set than what the tool actually told the model.
        """
        policies_json = json.dumps(policies or [], separators=(",", ":"), ensure_ascii=False)

        @tool
        def get_policies() -> str:
            """Return this request's AuthZEN policies as minified JSON: a list of
            {"policy": <policy definition>, "tags": [...]}. A policy's "actions" may be used
            as action.name when it applies to the resource in question (its resource.type
            matches) - mark such an evaluation with context.parameters.action_derivation_rule
            set to "policy_based".
            """
            return policies_json

        return get_policies

    def _build_tools(self, schema: dict[str, Any], policies: list[dict[str, Any]] | None) -> list[StructuredTool]:
        """Build this request's tool set, fresh each call from the exact schema/policies
        snapshot run_extraction already fetched once and built the response schema's enums,
        policy_context, and validate_evaluations from - every tool here returns that same
        snapshot rather than an independently re-fetched one (review), so what the LLM is
        shown, what gets validated, and what gets audited can never disagree.
        """
        return [
            t
            for t in (
                self._make_activate_skill_tool(policies),
                self._make_get_data_schema_tool(schema),
                self._make_get_policies_tool(policies),
            )
            if t
        ]

    def _make_activate_skill_tool(self, policies: list[dict[str, Any]] | None) -> StructuredTool | None:
        """Create activate_skill; extract-intent injects the data schema and, when available,
        this request's policies.

        `policies` is the exact snapshot run_extraction already fetched once and built
        request.policy_context / the response_schema's action enum from - not fetched again
        here. A second independent fetch could otherwise diverge from that snapshot (e.g. a
        first fetch that failed, followed by a successful retry moments later): the LLM would
        see policies that its own structured-output enum was never widened for, and that
        validate_evaluations/the audit trail never learn about either, so a genuinely
        policy-derived action could get wrongly rejected as off-schema, or the response could
        claim policy_fetch_failed when a later attempt actually succeeded. Reusing one
        snapshot for the whole request keeps what the LLM is told, what gets validated, and
        what gets audited all in agreement.
        """
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
                policy_block = _policy_block(policies, self.settings.policy_preview_chars)
                return (
                    f"<skill_content name={name!r}>\n"
                    f"{schema_block}{policy_block}\n\n{body}\n\n"
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

    async def _dispatch_tool_call(
        self,
        call: dict[str, Any],
        messages: list[BaseMessage],
        tools: list[StructuredTool],
        *,
        extract_intent_active: bool = False,
    ) -> bool:
        """Invoke one tool call, append ToolMessage, report if extract-intent is now active.

        A repeat activate_skill for an already-active extract-intent gets a short "answer now"
        reminder instead of the full skill body plus schema again: re-injecting those every
        round bloats the context and is what an LLM stuck re-activating the skill until
        TOOL_CALL_MAX_ITERATIONS runs out looks like in practice.

        Uses ainvoke, not invoke (review): every tool here (activate_skill, get_data_schema,
        get_policies) can call into a synchronous SchemaClient/PolicyClient - StructuredTool's
        own ainvoke() runs a plain sync-defined tool via a thread-pool executor rather than
        directly on the event loop, so a slow/hung fetch triggered by an on-demand tool call
        can no longer block every other concurrent request either, the same fix already
        applied to the schema/policy pre-checks in run_extraction.
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

        Schema/policy fetch, both moved inside REQUEST_TIMEOUT_SECONDS below, always run via
        `_run_in_fetch_executor` (see its docstring) rather than directly - including a cache
        hit, which is a synchronous call that can be real Redis network I/O depending on
        CACHE_BACKEND, not a bounded in-process lookup (review) - folding the fetch into the
        same deadline the LLM calls already share. Python cannot preempt an already-running
        synchronous call, so *this* request's own worst-case latency still tracks that client's
        own timeout x retry settings once a fetch has actually started, not a hard cutoff at
        REQUEST_TIMEOUT_SECONDS.

        `version`/`policy_context` are set before the `try` so the `except TimeoutError` block
        below has values even when the timeout fires mid-fetch, before either is assigned inside
        it. `policy_context`'s default has `fetch_failed=True` (review): the real PolicyContext
        further down only replaces it once get_cached_policies() actually returns, so a timeout
        firing while that fetch is still in flight must fall back to a default that already
        says the fetch failed, not one that looks like a successful fetch that just found no
        policies.

        `phase` tracks which of the three sections below was underway when a TimeoutError is
        caught, so `_timeout_result` can tell a slow schema/policy fetch from a slow LLM
        instead of blaming "the model" for either (review, ENG-9671) - see that method's
        docstring for the full reasoning.
        """
        version = ""
        policy_context = PolicyContext(policies=None, fetch_failed=True)
        phase = "schema fetch"
        try:
            async with asyncio.timeout(self.settings.request_timeout_seconds) as deadline:
                schema = await self._run_in_fetch_executor(self.get_cached_schema)
                if schema is None:
                    logger.error(
                        "Schema unavailable - cannot verify subject_type %r; skipping extraction for prompt %r",
                        subject.type,
                        prompt,
                    )
                    return ExtractionResult.failed(Outcome.ERROR, schema_unavailable_error(prompt))
                version = schema_version(schema)
                if schema_has_no_node_types(schema):
                    logger.warning(
                        "Schema fetched but the data model isn't populated yet - cannot verify "
                        "subject_type %r; skipping extraction for prompt %r",
                        subject.type,
                        prompt,
                    )
                    return ExtractionResult.failed(
                        Outcome.ERROR, schema_not_populated_error(prompt), schema_version=version
                    )

                try:
                    validate_subject_type(schema, subject.type)
                except SubjectTypeMismatchError as exc:
                    logger.error(  # noqa: TRY400
                        "subject_type %r invalid for the current schema; skipping extraction for prompt %r: %s",
                        subject.type,
                        prompt,
                        exc,
                    )
                    return ExtractionResult.failed(
                        Outcome.ERROR, subject_type_mismatch_error(prompt), schema_version=version
                    )

                phase = "policy fetch"
                policies = await self._run_in_fetch_executor(self.get_cached_policies, subject.type)
                policy_fetch_failed = policies is None
                policy_context = PolicyContext(policies=policies, fetch_failed=policy_fetch_failed)
                if policy_fetch_failed:
                    logger.warning(
                        "Policies unavailable for subject_type %r after retries; proceeding with "
                        "schema-only action derivation for prompt %r",
                        subject.type,
                        prompt,
                    )
                tools = self._build_tools(schema, policies)
                llm_with_tools = self._llm.bind_tools(
                    tools,
                    response_mime_type="application/json",
                    response_schema=build_input_schema(schema, policies),
                )
                request = _RequestContext(tools=tools, policy_context=policy_context)
                phase = "LLM interaction"
                return await self._extract_with_retries(
                    llm_with_tools,
                    prompt,
                    subject=subject,
                    schema=schema,
                    request=request,
                )
        except TimeoutError:
            return self._timeout_result(phase, prompt, deadline, version, policy_context)

    def _timeout_result(
        self,
        phase: str,
        prompt: str,
        deadline: asyncio.Timeout,
        version: str,
        policy_context: PolicyContext,
    ) -> ExtractionResult:
        """Build run_extraction's response to a TimeoutError, classified by which of its three
        phases (schema fetch, policy fetch, LLM interaction) was underway when it fired (review,
        ENG-9671) - see run_extraction's docstring for why that classification exists at all.

        Only the LLM-interaction phase uses llm_timeout_error ("The model did not respond in
        time.") - that text would misattribute a slow schema or policy fetch to the model, so
        those two use request_timeout_error instead. A schema-fetch-phase timeout also omits
        schema_version/policy_context, the same "not applicable yet" shape
        schema_unavailable_error's own path in run_extraction uses: neither a schema nor a
        policy was ever confirmed one way or the other. A policy-fetch-phase timeout reports
        both, since the schema is already known and policy_context's own `fetch_failed=True`
        default already correctly says that fetch was never completed.
        """
        if phase != "LLM interaction":
            logger.error(
                "%s exceeded the %.1fs end-to-end budget; giving up on prompt %r",
                phase.capitalize(),
                self.settings.request_timeout_seconds,
                prompt,
            )
            if phase == "schema fetch":
                return ExtractionResult.failed(Outcome.ERROR, request_timeout_error(prompt))
            return ExtractionResult.failed(
                Outcome.ERROR,
                request_timeout_error(prompt),
                schema_version=version,
                policy_context=policy_context,
            )
        # Both the per-call and the whole-request limits raise TimeoutError; only the
        # request one marks its own context as expired.
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
                if await self._dispatch_tool_call(call, messages, tools, extract_intent_active=extract_intent_active):
                    extract_intent_active = True
        return None, extract_intent_active

    async def _extract_with_retries(
        self,
        llm: Runnable[Any, BaseMessage],
        prompt: str,
        *,
        subject: RequestSubject,
        schema: dict[str, Any],
        request: _RequestContext,
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

        Recomputes schema_version(schema) rather than taking it as a separate parameter - the
        caller already needs it for its own fail-fast paths, but threading it through here too
        would push this method's arguments past the lint limit for no real benefit; hashing the
        schema once per request is negligible.
        """
        version = schema_version(schema)
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
            if response is None:
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
            final_text = self._response_text(response)
            if not activated_extract_intent:
                return ExtractionResult.free_text(
                    final_text,
                    schema_version=version,
                    policy_context=request.policy_context,
                )
            try:
                normalized = normalize_intent_response(
                    final_text,
                    prompt,
                    schema=schema,
                    subject=subject,
                    policy_context=request.policy_context,
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
                return ExtractionResult.from_normalized(
                    normalized,
                    schema_version=version,
                    validation_retries=retries,
                    policy_context=request.policy_context,
                )
        logger.error("Extraction still off-schema after %d retries; giving up on prompt %r", max_retries, prompt)
        return ExtractionResult.failed(
            Outcome.VALIDATION_FAILED,
            no_intent_error(prompt),
            schema_version=version,
            validation_retries=max_retries,
            policy_context=request.policy_context,
        )
