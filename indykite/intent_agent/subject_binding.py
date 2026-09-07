"""Schema-derived validation of the LLM's output (TDD 4.8) and deterministic subject binding (TDD 4.7)."""

import json
import re
from typing import Any

from indykite.intent_agent.errors import IntentValidationError, no_intent_error


def _node_types(schema: dict[str, Any]) -> set[str]:
    """Return every node type label defined in the given schema."""
    nodes = schema.get("graph", {}).get("nodes", {})
    if not isinstance(nodes, dict):
        return set()
    return set(nodes.keys())


def _edge_relations(schema: dict[str, Any]) -> set[str]:
    """Return every distinct edge relation (action name) defined in the given schema.

    Only non-empty strings count - a malformed relation (e.g. a number) is skipped, the same
    rule as input_schema._edge_relations so the enum and this check agree on the vocabulary.
    """
    edges = schema.get("graph", {}).get("edges", [])
    if not isinstance(edges, list):
        return set()
    return {
        relation
        for edge in edges
        if isinstance(edge, dict) and isinstance(relation := edge.get("relation"), str) and relation
    }


_BOOTSTRAP_NODE_TYPE = "_Application"


def schema_has_no_node_types(schema: dict[str, Any]) -> bool:
    """True if the schema has no node types beyond the platform's own bootstrap node.

    _Application (the calling app itself) is always present in the graph, even before any
    real data model is configured - so its presence alone doesn't mean the data model has
    been populated. Ignoring it, an empty remainder means nothing has been modeled yet -
    callers should treat this as a distinct "not yet populated" state, not a fetch failure.
    """
    return not (_node_types(schema) - {_BOOTSTRAP_NODE_TYPE})


def _blank(value: object) -> bool:
    """True unless value is a string with non-whitespace content."""
    return not isinstance(value, str) or not value.strip()


def _vocabulary_reason(field: str, value: object, allowed: set[str]) -> str | None:
    """Describe why `value` isn't acceptable for `field`, or None if it is. One line of the
    `reason` in errors.validation_feedback_error - not an error response itself.

    A blank value means the LLM tried and failed to correlate the field to the schema; a
    non-blank value outside `allowed` is the post-assembly schema-membership check itself
    (TDD 4.8). An empty `allowed` set skips the membership check - the same no-op rule as
    input_schema._constrain_to_enum, since a schema with nodes but no edges yet leaves
    action.name legitimately unconstrained.
    """
    if _blank(value):
        return f"{field} is missing or blank"
    if allowed and value not in allowed:
        return f"{field} {value!r} is not defined in the data schema"
    return None


def _evaluation_errors(evaluation: object, node_types: set[str], edge_relations: set[str]) -> list[str]:
    """Return every reason this evaluation can't be accepted, empty if it can.

    resource and subject are mandatory in every evaluation kind: resource.type is core to
    every AuthZEN evaluation (PRD), and subject.type is now the LLM's own call - it decides
    which node type represents the subject (the caller's identity for every kind but Subject
    Search, where it's the type being searched for - see SKILL.md), so an evaluation without
    one is malformed, not a legitimate omission. action is the one field still optional at
    the key level: Action Search omits it entirely (SKILL.md) - a key present but malformed
    (wrong type, or a blank/off-schema name inside it) means the LLM tried and failed to
    correlate it. subject.id is untouched here - _bind_subject binds it deterministically
    from the validated token afterward, regardless of what the LLM produced.
    """
    if not isinstance(evaluation, dict):
        return ["evaluation is not an object"]
    errors: list[str] = []
    resource = evaluation.get("resource")
    if not isinstance(resource, dict):
        errors.append("resource is missing or not an object")
    else:
        errors.append(_vocabulary_reason("resource.type", resource.get("type"), node_types))
    subject = evaluation.get("subject")
    if not isinstance(subject, dict):
        errors.append("subject is missing or not an object")
    else:
        errors.append(_vocabulary_reason("subject.type", subject.get("type"), node_types))
    action = evaluation.get("action")
    if action is not None:
        if not isinstance(action, dict):
            errors.append("action is not an object")
        else:
            errors.append(_vocabulary_reason("action.name", action.get("name"), edge_relations))
    return [error for error in errors if error]


def validate_evaluations(schema: dict[str, Any], evaluations: list[object]) -> list[str]:
    """Check every evaluation's subject.type / action.name / resource.type against the
    schema's node types and edge relations (TDD 4.8 post-assembly check). Returns one
    message per problem, prefixed with the evaluation's index so the LLM can tell which
    entry to fix (they become the `reason` lines of errors.validation_feedback_error);
    empty means the list is acceptable.
    """
    node_types = _node_types(schema)
    edge_relations = _edge_relations(schema)
    return [
        f"evaluations[{index}]: {error}"
        for index, evaluation in enumerate(evaluations)
        for error in _evaluation_errors(evaluation, node_types, edge_relations)
    ]


def _parse_llm_json(llm_text: str) -> object:
    """Parse the LLM's text as JSON (any JSON value), unwrapping a ```json fence first if present."""
    text = llm_text.strip()
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        text = json_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        message = f"the response is not valid JSON ({exc.msg} at position {exc.pos})"
        raise IntentValidationError([message]) from exc


def normalize_intent_response(
    llm_text: str,
    original_prompt: str,
    *,
    schema: dict[str, Any],
    subject_id: str | None,
) -> str:
    """Ensure the response is valid JSON with prompt/evaluations, wrap it if needed.

    Assumes the schema is available - run_extraction checks that upfront and never
    reaches this far (or the LLM) otherwise. The LLM's output is constrained to
    {prompt, evaluations}; an empty evaluations list is the LLM's legitimate "nothing
    coherent" signal (SKILL.md) and becomes no_intent_error without a retry. Anything
    else that can't be accepted - not JSON, or an off-schema/blank subject.type /
    action.name / resource.type - raises IntentValidationError so run_extraction can
    feed the errors back to the LLM and retry (TDD 4.8, 4.9).
    """
    data = _parse_llm_json(llm_text)
    if isinstance(data, list):
        # A bare JSON array is a plausible LLM mistake (forgetting the {"prompt", "evaluations"}
        # wrapper) - use its items as the evaluations directly instead of nesting the whole
        # array as a single non-dict "evaluation" that _bind_subject would then skip.
        data = {"prompt": original_prompt, "evaluations": data}
    if not isinstance(data, dict):
        raise IntentValidationError(["the response is not a JSON object"])
    data.setdefault("prompt", original_prompt)
    evaluations = data.get("evaluations")
    # Only an actual empty array is the "nothing coherent" signal - any other falsy or
    # non-list value ({}, 0, "", null, missing) is malformed output to retry, not a no-intent.
    if not isinstance(evaluations, list):
        raise IntentValidationError(["evaluations is missing or not an array"])
    if not evaluations:
        return json.dumps(no_intent_error(original_prompt), indent=2, ensure_ascii=False)
    errors = validate_evaluations(schema, evaluations)
    if errors:
        raise IntentValidationError(errors)

    _bind_subject(data, subject_id=subject_id)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _bind_subject(data: dict[str, Any], *, subject_id: str | None) -> None:
    """Force subject.id on every evaluation from the validated token, not whatever the LLM
    produced - closes the prompt-injection surface where the LLM could otherwise set its own
    identity. subject.type is untouched here: the LLM decides it (validated against the
    schema's node types by validate_evaluations before this runs), not deployment config.

    TODO: Subject Search evaluations use subject.id as a search parameter's absence, not the
    caller's own identity, and must not get this override - deferred to a later task.
    """
    for evaluation in data.get("evaluations", []):
        if not isinstance(evaluation, dict):
            continue
        subject = evaluation.get("subject")
        if not isinstance(subject, dict):
            continue
        if subject_id:
            subject["id"] = subject_id
        else:
            subject.pop("id", None)
