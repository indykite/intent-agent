"""The two things the LLM is given that derive from the live IndyKite graph schema: the
response_json_schema constraining its structured output to the schema's actual node/edge
vocabulary (TDD 4.6, build_input_schema), and the compact vocabulary summary injected into the
extract-intent skill activation (render_schema_summary).
"""

from typing import Any

from pydantic import BaseModel, Field, JsonValue

_DIGITAL_TWIN_LABEL = "DigitalTwin"


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


def _edge_relations(schema: dict[str, Any]) -> list[str]:
    """Return every distinct edge relation (action name) in the schema, sorted.

    Only non-empty strings count - a malformed relation (e.g. a number) is skipped rather than
    put into a string enum, the same rule as subject_binding._edge_relations so the enum and
    the post-assembly check agree on the vocabulary.
    """
    edges = _graph(schema).get("edges", [])
    if not isinstance(edges, list):
        return []
    relations = {
        relation
        for edge in edges
        if isinstance(edge, dict) and isinstance(relation := edge.get("relation"), str) and relation
    }
    return sorted(relations)


def _policy_action_names(policies: list[dict[str, Any]] | None) -> list[str]:
    """Every distinct action name any fetched policy defines, sorted - every
    policy's actions list regardless of resource.type, since the enum is one global list shared
    by every evaluation in the response (JSON Schema can't make it conditional per-evaluation);
    subject_binding._policy_actions_for_resource does the precise per-resource-type,
    marked-only check post-assembly. Same shape rule as _edge_relations: only non-empty strings.
    """
    if not policies:
        return []
    names: set[str] = set()
    for entry in policies:
        policy = entry.get("policy") if isinstance(entry, dict) else None
        actions = policy.get("actions") if isinstance(policy, dict) else None
        if isinstance(actions, list):
            names.update(name for name in actions if isinstance(name, str) and name)
    return sorted(names)


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
    """The LLM's entire structured-output shape - deliberately just {evaluations}, no
    LLM-authored error shape. When the model can't produce a coherent evaluation, it returns an
    empty evaluations array; our own code (subject_binding.no_intent_error) is solely
    responsible for the error response. The response's `prompt` echo (TDD 3.2) is not the
    model's to produce either: subject_binding.normalize_intent_response sets it from the
    incoming prompt, so asking the model for it only cost output tokens on every call.
    """

    evaluations: list[_Evaluation]


def _constrain_to_enum(field_schema: dict[str, Any], values: list[str]) -> None:
    """Restrict a (possibly nullable) string field's schema to `values`, in place.

    No-op if `values` is empty. Only reachable for the action-name enum (_edge_relations,
    optionally widened by _policy_action_names) - edges can't be ingested before
    their endpoint nodes, and build_input_schema's caller (agent.run_extraction) already
    validates SUBJECT_TYPE against the schema's node types before calling this, so
    _node_types(schema) can never be empty here. A schema with nodes but no edges yet, and no
    policies fetched either, is a legitimate state, not a failure - there's no vocabulary to
    constrain the action name to, so leaving it as a plain string is correct.
    """
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


def _policy_summaries(policies: list[dict[str, Any]] | None) -> list[str]:
    """One `subject -ACTION-> resource` line per action a fetched policy grants, sorted - the
    same {"policy": {subject, actions, resource}} entries _policy_action_names reads.
    """
    lines: set[str] = set()
    for entry in policies or []:
        policy = entry.get("policy") if isinstance(entry, dict) else None
        if not isinstance(policy, dict):
            continue
        subject, resource, actions = policy.get("subject"), policy.get("resource"), policy.get("actions")
        subject_type = subject.get("type") if isinstance(subject, dict) else None
        resource_type = resource.get("type") if isinstance(resource, dict) else None
        if not (isinstance(subject_type, str) and isinstance(resource_type, str) and isinstance(actions, list)):
            continue
        lines.update(
            f"- {subject_type} -{name}-> {resource_type}" for name in actions if isinstance(name, str) and name
        )
    return sorted(lines)


def render_schema_summary(schema: dict[str, Any], policies: list[dict[str, Any]] | None = None) -> str:
    """The schema's vocabulary as the model needs it, a few KB instead of the raw JGF: every
    node type with its subject eligibility and property names, every edge as
    `source -RELATION-> target`, and - when the request has any - the actions its access
    policies grant, in the same form, as a section of their own so the model knows they are
    policy-derived. The raw document, pretty-printed, runs to tens of thousands of characters
    for a real graph and used to be injected truncated mid-JSON at SCHEMA_PREVIEW_CHARS - the
    model then either guessed or re-fetched the schema until the tool-call budget ran out. The
    full JGF stays available through the get_data_schema tool.
    """
    policy_lines = _policy_summaries(policies)
    policy_heading = (
        "Policy actions (valid action names too - defined in access policies, always policy-derived), "
        "as subject -ACTION-> resource:"
    )
    nodes = _graph(schema).get("nodes", {})
    node_lines = (
        [_node_summary(label, node) for label, node in sorted(nodes.items())] if isinstance(nodes, dict) else []
    )
    node_heading = (
        "Node types (a type marked DigitalTwin may be a Subject; any type may be a Resource), with their properties:"
    )
    return "\n".join(
        [
            node_heading,
            *(node_lines or ["- (none)"]),
            "",
            "Edges (valid action names), as source -ACTION-> target:",
            *(_edge_summaries(schema) or ["- (none)"]),
            *(["", policy_heading, *policy_lines] if policy_lines else []),
        ]
    )


def build_input_schema(schema: dict[str, Any], policies: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the response_json_schema constraining the LLM's extraction output to this
    schema's actual vocabulary (TDD 4.6). Access-blind: enums list every entity/action the
    schema defines regardless of the caller's permissions (PRD 2.3.1).

    action.name's enum additionally includes every action any fetched policy defines, in
    a single global list, since JSON Schema can't make an
    enum conditional per-evaluation; subject_binding.validate_evaluations enforces, per
    evaluation, that a policy-only action name is only accepted when that evaluation is marked
    action_derivation_rule="policy_based" and the policy's resource.type actually matches.
    """
    json_schema = _IntentInput.model_json_schema()
    defs = json_schema["$defs"]
    _constrain_to_enum(defs["_Subject"]["properties"]["type"], _node_types(schema))
    action_names = sorted({*_edge_relations(schema), *_policy_action_names(policies)})
    _constrain_to_enum(defs["_Action"]["properties"]["name"], action_names)
    _constrain_to_enum(defs["_Resource"]["properties"]["type"], _node_types(schema))
    return _drop_pydantic_only_keys(json_schema)
