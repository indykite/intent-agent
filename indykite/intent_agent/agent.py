"""A2A executor, skill discovery, and Gemini-backed intent extraction pipeline.

Ported from the ``iag-demo/intent_agent`` prototype (see the Intent Agent Technical
Design Document). This is a structural port for this repository's package layout and
config model, not a rewrite of the extraction pipeline itself.
"""

import base64
import json
import logging
import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import httpx
import yaml
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact
from a2a.utils.constants import DEFAULT_RPC_URL
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.exceptions import HTTPException

from indykite.intent_agent.config import Settings

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
_TOOL_CALL_MAX_ITERATIONS = 5
_SCHEMA_PREVIEW_CHARS = 12000
_FRONTMATTER_PARTS = 3
_JWT_PARTS = 3
_BASE64_PAD_MODULUS = 4


def _coerce_str_list(value: object, delimiter: str = ",") -> list[str]:
    """Normalize a YAML frontmatter value (list, delimited string, or absent) into a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [s.strip() for s in value.split(delimiter) if s.strip()]
    return []


def _parse_frontmatter(raw: str, location: Path) -> tuple[dict[str, Any], str] | None:
    """Split a SKILL.md's raw text into its parsed YAML frontmatter and body."""
    parts = re.split(r"^---\s*$", raw.strip(), maxsplit=2, flags=re.MULTILINE)
    if len(parts) < _FRONTMATTER_PARTS:
        logger.warning("Skill file %s has no valid frontmatter (--- ... ---)", location)
        return None
    try:
        meta = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        logger.warning("Skill file %s has invalid YAML frontmatter", location)
        return None
    if not isinstance(meta, dict):
        return None
    return meta, parts[2].strip()


def _build_skill_record(meta: dict[str, Any], body: str, location: Path) -> dict[str, Any] | None:
    """Build a skill record from parsed frontmatter and body, or None if required fields are missing."""
    name = meta.get("name") or meta.get("title")
    description = meta.get("description")
    if not name or not description:
        logger.warning("Skill file %s missing name or description", location)
        return None
    name = str(name).strip()
    description = str(description).strip()
    if not description:
        return None
    return {
        "name": name,
        "description": description,
        "location": str(location),
        "body": body,
        "tags": _coerce_str_list(meta.get("tags")),
        "examples": _coerce_str_list(meta.get("examples"), delimiter="\n"),
    }


def _parse_skill_file(location: Path) -> dict[str, Any] | None:
    """Parse a SKILL.md file: extract YAML frontmatter and body. Returns a skill record or None."""
    try:
        raw = location.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Could not read skill file %s", location)
        return None
    if not raw.strip():
        return None
    parsed = _parse_frontmatter(raw, location)
    if parsed is None:
        return None
    meta, body = parsed
    return _build_skill_record(meta, body, location)


def _discover_skills() -> list[dict[str, Any]]:
    """Discover skills under _SKILLS_DIR: subdirectories containing a SKILL.md."""
    skills: list[dict[str, Any]] = []
    if not _SKILLS_DIR.is_dir():
        logger.info("Skills directory not found: %s", _SKILLS_DIR)
        return skills
    seen: set[str] = set()
    try:
        entries = sorted(_SKILLS_DIR.iterdir())
    except OSError:
        logger.warning("Error scanning skills dir %s", _SKILLS_DIR)
        return skills
    for entry in entries:
        if entry.name.startswith(".") or entry.name in {"node_modules", ".git"}:
            continue
        if not entry.is_dir():
            continue
        skill_md = entry / "SKILL.md"
        if not skill_md.is_file():
            continue
        record = _parse_skill_file(skill_md)
        if not record:
            continue
        name = record["name"]
        if name in seen:
            logger.warning("Duplicate skill name %r, skipping %s", name, skill_md)
            continue
        seen.add(name)
        skills.append(record)
    skills.sort(key=lambda s: str(s["name"]))
    return skills


def _build_skill_catalog_prompt(skills: list[dict[str, Any]]) -> str:
    """Build the tier-1 skill catalog for the system prompt (progressive disclosure)."""
    if not skills:
        return ""
    lines = [
        "You are an intent extraction agent. Follow these instructions to produce AuthZEN-compliant output.",
        "Your goal is to extract the intent from the user's prompt and return a JSON object that can be "
        "used to evaluate the intent.",
        "To do this, use the activate_skill tool to load the full instructions for a skill, and the "
        "get_data_schema tool to get the data schema.",
        "First, get the data schema to understand the entities and relationships in the system.",
        "Then, use the activate_skill tool to load the full instructions for the extract-intent skill.",
        "The extract-intent skill uses the data schema to map entities and relationships to the "
        "AuthZEN-compliant output.",
        "The following skills provide specialized instructions for specific tasks.",
        "",
        "<available_skills>",
    ]
    lines.extend(
        f"  <skill><name>{skill['name']}</name><description>{skill['description']}</description></skill>"
        for skill in skills
    )
    lines.append("</available_skills>")
    return "\n".join(lines)


def _build_agent_skill_catalog(skills: list[dict[str, Any]]) -> list[AgentSkill]:
    """Map discovered skill records onto A2A AgentSkill entries for the agent card."""
    catalog = [
        AgentSkill(
            id=str(s["name"]).replace(" ", "-").lower()[:64].strip("-"),
            name=str(s["name"]).replace("-", " ").title(),
            description=s["description"],
            tags=s.get("tags") or ["intent", "nlp", "extraction"],
            examples=s.get("examples") or [],
            input_modes=["text/plain"],
            output_modes=["text/plain"],
        )
        for s in skills
    ]
    if catalog:
        return catalog
    return [
        AgentSkill(
            id="extract-intent",
            name="Extract Intent",
            description=(
                "Extract intent from a provided input prompt text. Returns a structured "
                "description of the user's goal, action, and key entities."
            ),
            tags=["intent", "nlp", "extraction"],
            examples=[],
            input_modes=["text/plain"],
            output_modes=["text/plain"],
        ),
    ]


def _get_sub_from_token(token: str) -> str | None:
    """Extract the sub claim from a JWT access token. Returns None if the token is invalid or has no sub."""
    if not token.strip():
        return None
    parts = token.split(".")
    if len(parts) != _JWT_PARTS:
        return None
    payload_b64 = parts[1]
    padding = _BASE64_PAD_MODULUS - len(payload_b64) % _BASE64_PAD_MODULUS
    if padding != _BASE64_PAD_MODULUS:
        payload_b64 += "=" * padding
    payload_b64 = payload_b64.replace("-", "+").replace("_", "/")
    try:
        payload_bytes = base64.b64decode(payload_b64)
        data = json.loads(payload_bytes)
    except (ValueError, json.JSONDecodeError):
        return None
    sub = data.get("sub")
    return str(sub) if sub is not None else None


def _get_access_token_from_context(context: RequestContext | None) -> str:
    """Extract the Bearer token from the Authorization header of the incoming request."""
    if not context or not context.call_context:
        return ""
    req_headers = context.call_context.state.get("headers") or {}
    auth = req_headers.get("authorization") or req_headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return auth.strip()


def _normalize_intent_response(llm_text: str, original_prompt: str) -> str:
    """Ensure the response is valid JSON with prompt and evaluations. Fix or wrap it if needed."""
    text = llm_text.strip()
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        text = json_match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return json.dumps(
            {"prompt": original_prompt, "evaluations": [], "raw_response": text[:500]},
            indent=2,
            ensure_ascii=False,
        )
    if not isinstance(data, dict):
        data = {"prompt": original_prompt, "evaluations": [data] if data else []}
    data.setdefault("prompt", original_prompt)
    data.setdefault("evaluations", [])
    return json.dumps(data, indent=2, ensure_ascii=False)


class _ActivateSkillArgs(BaseModel):
    """Arguments schema for the activate_skill tool call."""

    name: str


class IntentAgent:
    """Owns the LLM, discovered skills, schema cache, and tool bindings for one running instance."""

    def __init__(self, settings: Settings) -> None:
        """Build the LLM, discover skills, and bind tools for this agent instance."""
        self.settings = settings
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            temperature=0,
        )
        self._schema_cache: dict[str, Any] | None = None
        self._current_subject_id: ContextVar[str | None] = ContextVar("current_subject_id", default=None)

        self.discovered_skills = _discover_skills()
        self.system_prompt = _build_skill_catalog_prompt(self.discovered_skills)
        self.skill_registry = {s["name"]: s for s in self.discovered_skills}
        self.agent_skills = _build_agent_skill_catalog(self.discovered_skills)

        self.tools = [t for t in (self._make_activate_skill_tool(), self._make_get_data_schema_tool()) if t]
        self._llm_with_tools = self._llm.bind_tools(self.tools)

    def load_data_schema(self) -> dict[str, Any] | None:
        """Fetch the graph schema from the Indykite platform. Returns a JGF object or None."""
        base_url = self.settings.indykite_base_url
        app_agent_key = self.settings.ik_app_agent_key
        if not base_url or not app_agent_key:
            logger.warning("INDYKITE_BASE_URL or IK_APP_AGENT_KEY not set - data schema unavailable")
            return None
        url = f"{base_url.rstrip('/')}/data-schema/v1"
        headers = {"X-IK-ClientKey": app_agent_key}
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
        except httpx.HTTPError:
            logger.warning("Failed to load data schema from %s", url)
            return None
        logger.info("Loaded data schema from %s", url)
        return resp.json()

    def preload_schema(self) -> None:
        """Fetch and cache the schema once, at startup."""
        self._schema_cache = self.load_data_schema()

    def get_cached_schema(self) -> dict[str, Any] | None:
        """Return the cached schema, fetching it once on first access if not preloaded."""
        if self._schema_cache is None:
            self._schema_cache = self.load_data_schema()
        return self._schema_cache

    def _make_get_data_schema_tool(self) -> StructuredTool:
        """Build the get_data_schema tool bound to this agent's cached schema."""
        agent = self

        @tool
        def get_data_schema() -> str:
            """Return the cached graph schema from the Indykite platform as minified JSON Graph Format (JGF).

            Describes all Node types and Edge types with their properties. Node labels are entity types.
            Edge labels can be Actions or context. All entities can be Resources. Nodes with system_labels
            containing DigitalTwin can also be Subjects.
            """
            schema = agent.get_cached_schema()
            if schema is None:
                return (
                    "Data schema unavailable (INDYKITE_BASE_URL or IK_APP_AGENT_KEY not configured, or fetch failed)."
                )
            return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)

        return get_data_schema

    def _make_activate_skill_tool(self) -> StructuredTool | None:
        """Create the activate_skill tool. For extract-intent, injects the data schema and token subject."""
        if not self.skill_registry:
            return None
        valid_names = sorted(self.skill_registry.keys())
        description = (
            "Load the full instructions for a skill by name. Call this when a task matches a skill's "
            f"description. Valid skill names: {', '.join(valid_names)}."
        )

        def _invoke(name: str) -> str:
            """Return the requested skill's content, or an error message if the name is unknown."""
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
                subject_id = self._current_subject_id.get()
                if schema:
                    schema_block = (
                        "\n\n<data_schema>\n"
                        + json.dumps(schema, indent=2, ensure_ascii=False)[:_SCHEMA_PREVIEW_CHARS]
                        + "\n</data_schema>\n\nUse the above schema to map entity types."
                    )
                else:
                    schema_block = (
                        "\n\nNo data schema available. Use generic types "
                        "(e.g., subject type 'user', resource type 'record')."
                    )
                if subject_id:
                    subject_block = (
                        f"\n\n<subject_from_token>The access token's 'sub' claim is: {subject_id!r}. "
                        "Use this as subject.id for Evaluation, Resource Search, and Action Search. "
                        "For Subject Search, omit subject.id.</subject_from_token>"
                    )
                else:
                    subject_block = (
                        "\n\n<subject_from_token>No subject from token. For Subject Search omit subject.id. "
                        "For others, omit or infer from prompt.</subject_from_token>"
                    )
                return (
                    f"<skill_content name={name!r}>\n"
                    f"{schema_block}{subject_block}\n\n{body}\n\n"
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

    def build_agent_card(self) -> AgentCard:
        """Build the A2A agent card advertising this agent's interface and skills."""
        return AgentCard(
            name=self.settings.intent_agent_name,
            description=(
                "Intent agent that extracts intent from a prompt and returns AuthZEN-compliant "
                "evaluation or search request objects. Uses the data schema to map entities to "
                "Subject, Action, and Resource."
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
                    url=f"http://{self.settings.advertised_host}:{self.settings.intent_port}",
                    protocol_version="1.0",
                ),
            ],
            skills=self.agent_skills,
        )

    def build_initial_messages(self, user_prompt: str) -> list[BaseMessage]:
        """Build initial messages: tier-1 skill catalog only (progressive disclosure)."""
        if not self.system_prompt:
            return [HumanMessage(content=user_prompt)]
        return [
            HumanMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

    @staticmethod
    def _extract_final_text(response: BaseMessage, *, activated_extract_intent: bool, prompt: str) -> str:
        """Render the LLM's final (non-tool-call) response as text, normalizing it if extract-intent ran."""
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
            final_text = _normalize_intent_response(final_text, prompt)
        return final_text

    def _dispatch_tool_call(self, call: dict[str, Any], messages: list[BaseMessage]) -> bool:
        """Invoke one tool call, append its ToolMessage, and report whether extract-intent was activated."""
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
        """Run the LLM + tool-call loop for one prompt and return the final response text."""
        self._current_subject_id.set(subject_id)
        messages = self.build_initial_messages(prompt)
        final_text = ""
        activated_extract_intent = False

        for _ in range(_TOOL_CALL_MAX_ITERATIONS):
            response = await self._llm_with_tools.ainvoke(messages)
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                final_text = self._extract_final_text(
                    response,
                    activated_extract_intent=activated_extract_intent,
                    prompt=prompt,
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
        """Handle one A2A task: extract intent from the prompt and emit the result artifact."""
        access_token = _get_access_token_from_context(context)
        if not access_token:
            raise HTTPException(status_code=401, detail="Authorization required")

        subject_id_from_token = _get_sub_from_token(access_token)

        raw_text = ""
        if context.message:
            for part in context.message.parts or []:
                text = getattr(part, "text", None)
                if text:
                    raw_text += text
        logger.info("Received message for %s: %s", self._agent.settings.intent_agent_name, raw_text)

        prompt = raw_text or "(empty)"

        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.TASK_STATE_WORKING,
                    message=new_agent_text_message("Extracting intent..."),
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
        logger.info("Intent extracted for %s: %s", self._agent.settings.intent_agent_name, final_text)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle task cancellation by emitting a CANCELED status event."""
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_CANCELED),
            ),
        )


def build_app(settings: Settings) -> tuple[Starlette, IntentAgent]:
    """Construct the Starlette app and its backing IntentAgent for the given settings."""
    agent = IntentAgent(settings)
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
