"""A2A executor and Gemini-backed intent extraction pipeline.

Ported from the ``iag-demo/intent_agent`` prototype (see the Intent Agent Technical
Design Document). This is a structural port for this repository's package layout and
config model, not a rewrite of the extraction pipeline itself.
"""

import json
import logging
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
from langchain_core.tools import StructuredTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED

from indykite.intent_agent import auth, skills
from indykite.intent_agent.config import Settings
from indykite.intent_agent.errors import SchemaUnavailableError, schema_unavailable_error
from indykite.intent_agent.schema import SchemaClient
from indykite.intent_agent.subject_binding import normalize_intent_response, validate_subject_type

logger = logging.getLogger(__name__)

_SECONDS_PER_MINUTE = 60


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
        self._llm_with_tools = self._llm.bind_tools(self.tools)

    def preload_schema(self) -> None:
        """Fetch and cache the schema once, at startup."""
        self._schema_client.preload()
        schema = self.get_cached_schema()
        if schema is None:
            message = "Data schema unavailable at cold start, the process cannot proceed without it."
            raise SchemaUnavailableError(message)
        validate_subject_type(schema, self.settings.subject_type)

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
            are entity types. Edge labels can be Actions or context. All entities can be
            Resources. Nodes with system_labels containing DigitalTwin can also be
            Subjects.
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
    def _extract_final_text(
        response: BaseMessage,
        *,
        activated_extract_intent: bool,
        prompt: str,
        subject_id: str | None,
        subject_type: str,
    ) -> str:
        """Render the LLM's final response, normalizing it if extract-intent ran."""
        raw_response_text = getattr(response, "text", "") or ""
        raw_content = getattr(response, "content", "") or ""
        if isinstance(raw_content, list) and raw_content and not raw_response_text:
            parts: list[str] = []
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    parts.append(getattr(block, "text", ""))
            final_text = " ".join(parts) if parts else ""
        else:
            final_text = str(raw_response_text or raw_content or "")
        if activated_extract_intent:
            final_text = normalize_intent_response(
                final_text,
                prompt,
                subject_id=subject_id,
                subject_type=subject_type,
            )
        return final_text

    def _dispatch_tool_call(self, call: dict[str, Any], messages: list[BaseMessage]) -> bool:
        """Invoke one tool call, append ToolMessage, report if extract-intent ran."""
        name = call.get("name", "")
        args = call.get("args", {}) or {}
        call_id = call.get("id", "")
        activated_extract_intent = name == "activate_skill" and args.get("name") == "extract-intent"
        tool_obj = next((t for t in self.tools if t.name == name), None)
        if tool_obj:
            try:
                result = tool_obj.invoke(args)
            except Exception as exc:  # noqa: BLE001 - tool errors are fed back to the LLM, not raised
                result = f"Error: {exc}"
        else:
            result = "Tool not found"
        messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
        return activated_extract_intent

    async def run_extraction(self, prompt: str, subject_id: str | None) -> str:
        """Run the LLM tool-call loop for one prompt, return the final response text.

        Checks schema availability upfront, before any LLM call: subject.type can't be
        verified without it, so there's no point spending an LLM call (or several, across
        tool-call iterations) on a result that would just be discarded afterward.
        """
        if self.get_cached_schema() is None:
            logger.error(
                "Schema unavailable - cannot verify subject_type %r; skipping extraction for prompt %r",
                self.settings.subject_type,
                prompt,
            )
            return json.dumps(schema_unavailable_error(prompt), indent=2, ensure_ascii=False)

        messages = self.build_initial_messages(prompt)
        final_text = ""
        activated_extract_intent = False

        for _ in range(self.settings.tool_call_max_iterations):
            response = await self._llm_with_tools.ainvoke(messages)
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                final_text = self._extract_final_text(
                    response,
                    activated_extract_intent=activated_extract_intent,
                    prompt=prompt,
                    subject_id=subject_id,
                    subject_type=self.settings.subject_type,
                )
                break

            messages.append(response)
            for call in tool_calls:
                if self._dispatch_tool_call(call, messages):
                    activated_extract_intent = True

        return final_text or "(No response generated)"


class IntentExecutor(AgentExecutor):
    """AgentExecutor that extracts intent from incoming text using an LLM."""

    def __init__(self, agent: IntentAgent) -> None:
        """Wrap the given IntentAgent as an A2A AgentExecutor."""
        self._agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle one A2A task: extract intent, emit the result artifact."""
        access_token = auth.get_access_token_from_context(context)
        if not access_token:
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Authorization required")

        settings = self._agent.settings
        try:
            claims = await auth.validate_access_token(
                access_token,
                introspect_url=str(settings.introspect_url),
                client_id=settings.introspect_client_id,
                client_secret=settings.introspect_client_secret.get_secret_value(),
                timeout_seconds=settings.introspect_timeout_seconds,
            )
        except auth.TokenValidationError as exc:
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

        subject_id_from_token = claims["sub"]

        raw_text = ""
        if context.message:
            for part in context.message.parts or []:
                text = getattr(part, "text", None)
                if text:
                    raw_text += text
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

        final_text = await self._agent.run_extraction(prompt, subject_id_from_token)

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                artifact=new_text_artifact(name="result", text=final_text),
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
            final_text,
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


def build_app(
    settings: Settings,
    *,
    llm: BaseChatModel | None = None,
    schema_client: SchemaClient | None = None,
) -> tuple[Starlette, IntentAgent]:
    """Construct the Starlette app and its backing IntentAgent for these settings."""
    agent = IntentAgent(settings, llm=llm, schema_client=schema_client)
    agent.preload_schema()

    agent_card = agent.build_agent_card()
    handler = DefaultRequestHandler(
        agent_executor=IntentExecutor(agent),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    app = Starlette(
        routes=[
            *create_agent_card_routes(agent_card=agent_card),
            *create_jsonrpc_routes(request_handler=handler, rpc_url=DEFAULT_RPC_URL),
        ],
    )
    return app, agent
