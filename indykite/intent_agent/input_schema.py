"""Builds the schema-derived response_json_schema that constrains Gemini's structured output
to the live IndyKite graph schema's actual node/edge vocabulary (TDD 4.6).
"""

from typing import Any

from pydantic import BaseModel, Field, JsonValue


def _node_types(schema: dict[str, Any]) -> list[str]:
    """Return every node type label in the schema, sorted for a stable enum order."""
    nodes = schema.get("graph", {}).get("nodes", {})
    if not isinstance(nodes, dict):
        return []
    return sorted(nodes.keys())


def _edge_relations(schema: dict[str, Any]) -> list[str]:
    """Return every distinct edge relation (action name) in the schema, sorted.

    Only non-empty strings count - a malformed relation (e.g. a number) is skipped rather than
    put into a string enum, the same rule as subject_binding._edge_relations so the enum and
    the post-assembly check agree on the vocabulary.
    """
    edges = schema.get("graph", {}).get("edges", [])
    if not isinstance(edges, list):
        return []
    relations = {
        relation
        for edge in edges
        if isinstance(edge, dict) and isinstance(relation := edge.get("relation"), str) and relation
    }
    return sorted(relations)


class _Subject(BaseModel):
    """AuthZEN subject. `id` is deliberately absent - it's assigned deterministically
    post-LLM from the validated access token, never LLM-supplied.
    """

    type: str | None = None


class _Action(BaseModel):
    """AuthZEN action."""

    name: str | None = None


class _Resource(BaseModel):
    """AuthZEN resource."""

    id: str | None = None
    type: str | None = None


class _Context(BaseModel):
    """AuthZEN context: prompt-derived data that doesn't fit subject/action/resource."""

    parameters: dict[str, Any] | None = Field(
        default=None,
        description="Extra prompt-derived data that doesn't fit subject/action/resource.",
    )
    filters: str | None = Field(default=None, description="Search criteria, e.g. \"resource.name contains 'refund'\".")
    qualifiers: dict[str, Any] | None = Field(default=None, description="Environmental info, e.g. time or location.")


class _Evaluation(BaseModel):
    """A single AuthZEN evaluation entry."""

    subject: _Subject = Field(default_factory=_Subject)
    action: _Action = Field(default_factory=_Action)
    resource: _Resource = Field(default_factory=_Resource)
    context: _Context = Field(default_factory=_Context)


class _IntentInput(BaseModel):
    """The LLM's entire structured-output shape - deliberately just {prompt, evaluations}, no
    LLM-authored error shape. When the model can't produce a coherent evaluation, it returns an
    empty evaluations array; our own code (subject_binding.no_intent_error) is solely
    responsible for the error response.
    """

    prompt: str
    evaluations: list[_Evaluation]


def _constrain_to_enum(field_schema: dict[str, Any], values: list[str]) -> None:
    """Restrict a (possibly nullable) string field's schema to `values`, in place.

    No-op if `values` is empty. Only reachable for the action-name enum (_edge_relations) -
    edges can't be ingested before their endpoint nodes, and build_input_schema's caller
    (agent.run_extraction) already validates SUBJECT_TYPE against the schema's node types
    before calling this, so _node_types(schema) can never be empty here. A schema with nodes
    but no edges yet is a legitimate state, not a failure - there's no vocabulary to constrain
    the action name to, so leaving it as a plain string is correct.
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


def build_input_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Build the response_json_schema constraining the LLM's extraction output to this
    schema's actual vocabulary (TDD 4.6). Access-blind: enums list every entity/action the
    schema defines regardless of the caller's permissions (PRD 2.3.1).
    """
    json_schema = _IntentInput.model_json_schema()
    defs = json_schema["$defs"]
    _constrain_to_enum(defs["_Subject"]["properties"]["type"], _node_types(schema))
    _constrain_to_enum(defs["_Action"]["properties"]["name"], _edge_relations(schema))
    _constrain_to_enum(defs["_Resource"]["properties"]["type"], _node_types(schema))
    return _drop_pydantic_only_keys(json_schema)
