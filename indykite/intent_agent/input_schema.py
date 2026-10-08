"""What the LLM is given from the caller's policies and the graph schema: the structured-output
schema (build_input_schema, TDD 4.6) and the vocabulary summary for skill activation
(render_schema_summary).
"""

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, JsonValue

_DIGITAL_TWIN_LABEL = "DigitalTwin"

# The model's rating of how well the chosen policies cover the prompt (see SKILL.md).
Confidence = Literal["high", "medium", "low"]
CONFIDENCE_LEVELS: tuple[str, ...] = get_args(Confidence)


def _graph(schema: dict[str, Any]) -> dict[str, Any]:
    """The JGF `graph` object, or `{}` when the document has none or it isn't an object - every
    reader below tolerates a missing or malformed section the same way, so a malformed platform
    response reads as an empty graph rather than raising on `.get`.
    """
    graph = schema.get("graph")
    return graph if isinstance(graph, dict) else {}


def _node_types(schema: dict[str, Any]) -> list[str]:
    """Return every node type label in the schema, sorted for a stable enum order."""
    nodes = _graph(schema).get("nodes", {})
    if not isinstance(nodes, dict):
        return []
    return sorted(nodes.keys())


def _policy_type(policy: dict[str, Any], key: str) -> str | None:
    """The policy's subject.type / resource.type (`key`), or None when it isn't a non-empty string."""
    part = policy.get(key)
    value = part.get("type") if isinstance(part, dict) else None
    return value if isinstance(value, str) and value else None


def policy_triples(policies: list[dict[str, Any]] | None, schema: dict[str, Any]) -> set[tuple[str, str, str]]:
    """Every (subject type, action, resource type) the policies grant over the schema's node
    types - the whole vocabulary an evaluation may use. Malformed entries are skipped.
    """
    node_types = set(_node_types(schema))
    triples: set[tuple[str, str, str]] = set()
    for entry in policies or []:
        policy = entry.get("policy") if isinstance(entry, dict) else None
        if not isinstance(policy, dict):
            continue
        subject_type, resource_type = _policy_type(policy, "subject"), _policy_type(policy, "resource")
        actions = policy.get("actions")
        if subject_type not in node_types or resource_type not in node_types or not isinstance(actions, list):
            continue
        triples.update(
            (subject_type, name, resource_type) for name in actions if isinstance(name, str) and name.strip()
        )
    return triples


class _Subject(BaseModel):
    """AuthZEN subject. `id` is normally left absent - it's assigned deterministically
    post-LLM from the caller-supplied subject params, never trusted from the LLM. The
    exceptions are the two marked cases (SKILL.md): a second subject, where the LLM must set
    both fields explicitly, and a Subject Search, where it sets only `type` (the entity type
    searched for).
    """

    type: str | None = Field(
        default=None,
        description=(
            "Only for a Subject Search (the entity type being searched for - see the Subject "
            "Search section) or an evaluation about a second, specific subject identified in the "
            "prompt. Omit otherwise; the caller's own type is filled in automatically."
        ),
    )
    id: str | None = Field(
        default=None,
        description=(
            "Only for an evaluation about a second, specific subject identified in the prompt "
            "(not the caller) - see the Second Subject section. Omit otherwise; the caller's own "
            "identity is filled in automatically and any value given here is ignored."
        ),
    )


class _Action(BaseModel):
    """AuthZEN action."""

    name: str | None = None


class _Resource(BaseModel):
    """AuthZEN resource."""

    # Class docstrings and field descriptions reach Gemini as the schema's `description`s, so
    # they're written for the model. The `id` description steers it toward filling the id in
    # rather than restating the identifier as a context.filters term - seen live with flash-lite,
    # which still prefers the filter form more often than not (test/test_golden_scenarios.py
    # replays the answers it does give), and which drops the identifier altogether when told
    # more forcefully never to use the filter form; subject_binding lifts the pure external_id
    # filter form back into the id afterwards instead.
    # Required (not nullable): Gemini fills a required field far more reliably than an optional
    # one, and flash-lite dropped a person's identifier ("user flo") when it was optional. An
    # empty string is the "many resources" answer and is removed again by
    # subject_binding._drop_blank_resource_ids.
    id: str = Field(
        description=(
            "The identifier (external_id) of the one resource instance the prompt names, e.g. an "
            "account number, a document id, or a person's identifier ('flo' in 'user flo'). Use "
            'the empty string "" only when the prompt is about many resources rather than one.'
        ),
    )
    type: str | None = None


class _Context(BaseModel):
    """AuthZEN context: prompt-derived data that doesn't fit subject/action/resource."""

    parameters: dict[str, Any] | None = Field(
        default=None,
        description="Extra prompt-derived data that doesn't fit subject/action/resource.",
    )
    filters: str | None = Field(
        default=None,
        description=(
            "Search criteria as a CEL expression over the resource's properties, "
            "e.g. \"resource.name.contains('refund')\"."
        ),
    )
    qualifiers: dict[str, Any] | None = Field(default=None, description="Environmental info, e.g. time or location.")


class _Evaluation(BaseModel):
    """A single AuthZEN evaluation entry."""

    subject: _Subject = Field(default_factory=_Subject)
    action: _Action = Field(default_factory=_Action)
    resource: _Resource = Field(default_factory=_Resource)
    context: _Context = Field(default_factory=_Context)


class _IntentInput(BaseModel):
    """The LLM's whole structured output. No error shape or prompt echo - the system builds those.
    Gemini writes fields in order, so `explanation` comes first: the model names its policy
    before writing the evaluations.
    """

    # Optional so a refusal (empty evaluations) can leave them out; subject_binding requires both
    # on an answer with evaluations.
    explanation: str | None = Field(
        default=None,
        description=(
            "Written first, before the evaluations. One short, plain sentence: the policy you will "
            "use and, if any, what part of the prompt no policy covers."
        ),
    )
    evaluations: list[_Evaluation]
    confidence: Confidence | None = Field(
        default=None,
        description=(
            "high: the prompt matches the policies exactly. medium: it matches only through a synonym "
            "- a policy's resource type or action stands in for the one asked. low: only part of the "
            "prompt matched a policy, the rest was left out."
        ),
    )


def _constrain_to_enum(field_schema: dict[str, Any], values: list[str]) -> None:
    """Restrict a (possibly nullable) string field's schema to `values`, in place; no-op if empty."""
    if not values:
        return
    for variant in field_schema.get("anyOf", [field_schema]):
        if variant.get("type") == "string":
            variant["enum"] = values


def _drop_pydantic_only_keys(node: JsonValue) -> JsonValue:
    """Strip pydantic's auto-generated `title`/`default` keys recursively."""
    if isinstance(node, dict):
        return {key: _drop_pydantic_only_keys(value) for key, value in node.items() if key not in {"title", "default"}}
    if isinstance(node, list):
        return [_drop_pydantic_only_keys(item) for item in node]
    return node


def _has_digital_twin_label(node: object) -> bool:
    """True when a JGF node entry carries the DigitalTwin system label."""
    metadata = node.get("metadata", {}) if isinstance(node, dict) else {}
    system_labels = metadata.get("system_labels", []) if isinstance(metadata, dict) else []
    return isinstance(system_labels, list) and any(
        isinstance(entry, dict) and entry.get("name") == _DIGITAL_TWIN_LABEL for entry in system_labels
    )


def digital_twin_types(schema: dict[str, Any]) -> list[str]:
    """The node types that may be subjects (system label DigitalTwin), minus the platform's own
    `_`-prefixed bootstrap nodes (`_Application`), sorted - the one place this label walk lives,
    shared by the schema summary and the golden eval loader.
    """
    nodes = _graph(schema).get("nodes", {})
    if not isinstance(nodes, dict):
        return []
    return sorted(label for label, node in nodes.items() if not label.startswith("_") and _has_digital_twin_label(node))


def _node_summary(label: str, node: object) -> str:
    """One line for a node type: its label, whether it may be a subject, and its property names.
    A `_`-prefixed type is the platform's own (`_Application`, the calling app) - it carries the
    DigitalTwin label too but is never what a prompt is about, so it is flagged as internal
    rather than as a Subject candidate.
    """
    metadata = node.get("metadata", {}) if isinstance(node, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    digital_twin = _has_digital_twin_label(node)
    properties = metadata.get("properties", {})
    names = sorted(properties) if isinstance(properties, dict) else []
    if label.startswith("_"):
        kind = " (platform-internal, never a Subject or Resource)"
    else:
        kind = f" ({_DIGITAL_TWIN_LABEL})" if digital_twin else ""
    return f"- {label}{kind}: {', '.join(names) if names else '(no properties)'}"


def _edge_summaries(schema: dict[str, Any]) -> list[str]:
    """One `source -RELATION-> target` line per distinct well-formed edge, sorted."""
    edges = _graph(schema).get("edges", [])
    if not isinstance(edges, list):
        return []
    lines = {
        f"- {edge['source']} -{edge['relation']}-> {edge['target']}"
        for edge in edges
        if isinstance(edge, dict)
        and all(isinstance(edge.get(key), str) and edge.get(key) for key in ("source", "relation", "target"))
    }
    return sorted(lines)


def render_schema_summary(schema: dict[str, Any], policies: list[dict[str, Any]] | None = None) -> str:
    """The request's vocabulary in a few KB instead of the raw JGF: node types with their
    properties, edges (how data is linked, never actions) and the combinations the policies grant.
    """
    nodes = _graph(schema).get("nodes", {})
    node_lines = (
        [_node_summary(label, node) for label, node in sorted(nodes.items())] if isinstance(nodes, dict) else []
    )
    policy_lines = sorted(
        f"- {subject_type} -{action}-> {resource_type}"
        for subject_type, action, resource_type in policy_triples(policies, schema)
    )
    return "\n".join(
        [
            "Node types, with their properties (only a type a policy below names may be a Subject or Resource):",
            *(node_lines or ["- (none)"]),
            "",
            "Edges (how the data is linked - never action names), as source -RELATION-> target:",
            *(_edge_summaries(schema) or ["- (none)"]),
            "",
            "Policies (the only valid combinations - every evaluation uses one), as subject -ACTION-> resource:",
            *(policy_lines or ["- (none)"]),
        ]
    )


def build_input_schema(schema: dict[str, Any], policies: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The structured-output schema, with subject.type, action.name and resource.type each limited
    to its part of policy_triples. JSON Schema can't tie the three together; validate_evaluations does.
    """
    triples = policy_triples(policies, schema)
    json_schema = _IntentInput.model_json_schema()
    defs = json_schema["$defs"]
    _constrain_to_enum(defs["_Subject"]["properties"]["type"], sorted({subject for subject, _, _ in triples}))
    _constrain_to_enum(defs["_Action"]["properties"]["name"], sorted({action for _, action, _ in triples}))
    _constrain_to_enum(defs["_Resource"]["properties"]["type"], sorted({resource for _, _, resource in triples}))
    return _drop_pydantic_only_keys(json_schema)
