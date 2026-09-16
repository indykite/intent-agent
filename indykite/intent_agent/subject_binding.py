"""Schema-derived validation of the LLM's output (TDD 4.8) and deterministic subject binding (TDD 4.7)."""

import json
import re
from typing import Any

from indykite.intent_agent.errors import (
    SCHEMA_BLANK_SUFFIX,
    SCHEMA_MEMBERSHIP_SUFFIX,
    IntentValidationError,
    SubjectTypeMismatchError,
    no_intent_error,
)


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
    been populated. Ignoring it, an empty remainder means nothing has been modeled yet, not
    that SUBJECT_TYPE itself is misconfigured (SubjectTypeMismatchError, raised when at least
    one *other* node type exists and SUBJECT_TYPE just isn't one of them) - callers should
    treat this as a distinct "not yet populated" state, not a fetch failure or a mismatch.
    """
    return not (_node_types(schema) - {_BOOTSTRAP_NODE_TYPE})


def validate_subject_type(schema: dict[str, Any] | None, subject_type: str) -> None:
    """Raise if subject_type isn't one of the schema's node types."""
    if schema is None:
        return
    types = _node_types(schema)
    if subject_type not in types:
        message = f"SUBJECT_TYPE {subject_type!r} is not a node type in the schema."
        raise SubjectTypeMismatchError(message)


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
        return f"{field} {SCHEMA_BLANK_SUFFIX}"
    if allowed and value not in allowed:
        return f"{field} {value!r} {SCHEMA_MEMBERSHIP_SUFFIX}"
    return None


# The PRD's context sub-shapes (TDD 3.2) and what each must be when present - the same three
# keys input_schema._Context declares for the structured output.
_CONTEXT_KEY_TYPES: dict[str, type] = {"filters": str, "parameters": dict, "qualifiers": dict}
_TYPE_NAMES = {str: "a string", dict: "an object"}


def _context_errors(context: object) -> list[str]:
    """Describe every way `context` departs from the PRD's shape (TDD 3.2): an object whose
    keys have the types in _CONTEXT_KEY_TYPES. An absent or null context, or an absent/null
    key, is fine - each is optional; any other key is passed through untouched (the PRD calls
    context an open object). The structured output already types these, but like the
    type/name enums that's a strong hint to Gemini, not a guarantee, so this is the same
    belt-and-braces check as _vocabulary_reason.
    """
    if context is None:
        return []
    if not isinstance(context, dict):
        return ["context is not an object"]
    return [
        f"context.{key} is not {_TYPE_NAMES[expected]}"
        for key, expected in _CONTEXT_KEY_TYPES.items()
        if (value := context.get(key)) is not None and not isinstance(value, expected)
    ]


def _evaluation_errors(evaluation: object, node_types: set[str], edge_relations: set[str]) -> list[str]:
    """Return every reason this evaluation can't be accepted, empty if it can.

    Must run before _bind_subject, which unconditionally overwrites subject.type/id
    regardless of what the LLM produced. A key entirely absent (e.g. no `subject` for
    Access Evaluation, no `action` for Action Search - see SKILL.md) is a legitimate
    per-evaluation-kind omission, not a failure; a key present but malformed (wrong
    type, or a blank/off-schema type/name inside it) means the LLM tried and failed to
    correlate it. subject is exempt from the malformed-shape check (though not the
    blank/off-schema one) since _bind_subject replaces it outright either way. context
    has no schema vocabulary to check against, only a shape (_context_errors).
    """
    if not isinstance(evaluation, dict):
        return ["evaluation is not an object"]
    errors: list[str] = []
    resource = evaluation.get("resource")
    if not isinstance(resource, dict):
        errors.append("resource is missing or not an object")
    else:
        errors.append(_vocabulary_reason("resource.type", resource.get("type"), node_types))
    action = evaluation.get("action")
    if action is not None:
        if not isinstance(action, dict):
            errors.append("action is not an object")
        else:
            errors.append(_vocabulary_reason("action.name", action.get("name"), edge_relations))
    subject = evaluation.get("subject")
    if isinstance(subject, dict):
        errors.append(_vocabulary_reason("subject.type", subject.get("type"), node_types))
    errors.extend(_context_errors(evaluation.get("context")))
    return [error for error in errors if error]


def validate_evaluations(schema: dict[str, Any], evaluations: list[object]) -> list[str]:
    """Check every evaluation's subject.type / action.name / resource.type against the
    schema's node types and edge relations (TDD 4.8 post-assembly check), and its context
    against the PRD's parameters/filters/qualifiers shape. Returns one message per problem,
    prefixed with the evaluation's index so the LLM can tell which entry to fix (they
    become the `reason` lines of errors.validation_feedback_error); empty means the list
    is acceptable.
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
    subject_type: str,
) -> str:
    """Ensure the response is valid JSON with an evaluations list, wrap it if needed, and set
    the prompt echo.

    Assumes the schema is available - run_extraction checks that upfront and never
    reaches this far (or the LLM) otherwise. The LLM's output is constrained to
    {evaluations} (a `prompt` it adds anyway is ignored); an empty evaluations list is the
    LLM's legitimate "nothing coherent" signal (SKILL.md) and becomes no_intent_error
    without a retry. Anything
    else that can't be accepted - not JSON, an off-schema/blank subject.type /
    action.name / resource.type, or a mis-shaped context - raises IntentValidationError
    so run_extraction can feed the errors back to the LLM and retry (TDD 4.8, 4.9).
    """
    data = _parse_llm_json(llm_text)
    if isinstance(data, list):
        # A bare JSON array is a plausible LLM mistake (forgetting the {"evaluations"}
        # wrapper) - use its items as the evaluations directly instead of nesting the whole
        # array as a single non-dict "evaluation" that _bind_subject would then skip.
        data = {"prompt": original_prompt, "evaluations": data}
    if not isinstance(data, dict):
        raise IntentValidationError(["the response is not a JSON object"])
    # `prompt` echoes the incoming prompt verbatim (TDD 3.2) - set here, never taken from the
    # LLM (it isn't part of the structured output any more; when it was, an injection-style
    # prompt came back "helpfully" shortened, minus the part carrying the injected values).
    # Rebuilt rather than assigned so it leads the response the way the TDD shows it.
    data = {"prompt": original_prompt, **{key: value for key, value in data.items() if key != "prompt"}}
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

    _bind_subject(data, subject_id=subject_id, subject_type=subject_type)
    _normalize_contexts(data)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _normalize_contexts(data: dict[str, Any]) -> None:
    """Settle every evaluation's context into the PRD shape after validation: always an
    object (`{}` when the LLM gave none), with the structured output's nullability
    artefacts removed - a null key, or a blank `filters` string, both mean "not
    applicable", and the caller shouldn't have to tell those apart from an absent key.
    Everything else is passed through verbatim: context is user-controlled data, and it's
    the downstream consumer's job to treat it as untrusted (TDD 9), not this agent's to
    rewrite it.
    """
    for evaluation in data.get("evaluations", []):
        if not isinstance(evaluation, dict):
            continue
        context = evaluation.get("context")
        if not isinstance(context, dict):
            context = {}
        # Only the three contract keys are nullable in the structured output, so only their
        # nulls are artefacts; any other key is passed through exactly as the model wrote it.
        context = {key: value for key, value in context.items() if value is not None or key not in _CONTEXT_KEY_TYPES}
        if "filters" in context and _blank(context["filters"]):
            del context["filters"]
        evaluation["context"] = context


# SKILL.md's marker for "this evaluation's subject refers to a second subject the LLM
# identified in the prompt, not the caller" - set in context.parameters, the one part of an
# evaluation already documented as free-form prompt-derived data. Its absence is the default
# and safe fallback: override, exactly as when there is no second subject at all.
_SUBJECT_SOURCE_KEY = "subject_source"
_SUBJECT_SOURCE_PROMPT = "prompt"


def _gate_resource(data: dict[str, Any]) -> tuple[str, str] | None:
    """Return (type, id) of evaluations[0]'s resource - the caller-anchored gate evaluation the
    second-subject pattern hinges on (SKILL.md: "resource is the other subject itself") - or
    None if evaluations[0] isn't shaped that way. The marker alone (_subject_from_prompt) is
    never enough to trust a claimed subject; it must also match the one entity the caller
    actually gated in evaluations[0], not just claim to.
    """
    evaluations = data.get("evaluations", [])
    if not evaluations or not isinstance(evaluations[0], dict):
        return None
    resource = evaluations[0].get("resource")
    if not isinstance(resource, dict):
        return None
    resource_type, resource_id = resource.get("type"), resource.get("id")
    if not isinstance(resource_type, str) or not isinstance(resource_id, str):
        return None
    return resource_type, resource_id


def _subject_from_prompt(evaluation: dict[str, Any], gate_resource: tuple[str, str] | None) -> bool:
    """True if the LLM marked this evaluation's subject as the second subject it identified
    in the prompt (see SKILL.md), rather than the caller, *and* that subject matches
    evaluations[0]'s gate resource - the one case _bind_subject must leave alone instead of
    overriding. The marker by itself is not enough: an evaluation marked for a subject the
    caller never actually gated in evaluations[0] falls back to the safe default (override),
    same as an absent or malformed marker - otherwise the marker alone would let the LLM vouch
    for an arbitrary schema-valid subject with no real gate behind it at all.
    """
    context = evaluation.get("context")
    if not isinstance(context, dict):
        return False
    parameters = context.get("parameters")
    if not isinstance(parameters, dict):
        return False
    if parameters.get(_SUBJECT_SOURCE_KEY) != _SUBJECT_SOURCE_PROMPT:
        return False
    if gate_resource is None:
        return False
    subject = evaluation.get("subject")
    if not isinstance(subject, dict):
        return False
    return (subject.get("type"), subject.get("id")) == gate_resource


def _bind_subject(data: dict[str, Any], *, subject_id: str | None, subject_type: str) -> None:
    """Force subject.id/subject.type on every evaluation from the caller-supplied subject
    params (IntentExecutor._subject_params - resolved by IAG per request, not a token claim
    or static deployment config), not whatever the LLM produced - closes the prompt-injection
    surface where the LLM could otherwise set the subject itself.

    The one exception: an evaluation *after the first* that the LLM explicitly marked as being
    about a second subject it identified in the prompt, and whose subject matches evaluations[0]'s
    gate resource (_subject_from_prompt), keeps the subject it produced - trusted for its type
    (still schema-validated by validate_evaluations) and its id (free text, like resource.id).
    Every other evaluation - the normal single-subject case, any evaluation the LLM didn't mark
    or marked for a subject the gate doesn't cover, and evaluations[0] no matter what - gets the
    override regardless of what the LLM put there, same as before.

    evaluations[0] is deliberately *never* eligible for the exception, marked or not: it is the
    caller-anchored evaluation the whole pattern hinges on, so a response with only one
    evaluation (no second subject at all) must always come back bound to the caller, and can
    never be trusted on the LLM's say-so alone.
    """
    gate_resource = _gate_resource(data)
    for index, evaluation in enumerate(data.get("evaluations", [])):
        if not isinstance(evaluation, dict):
            continue
        if index > 0 and _subject_from_prompt(evaluation, gate_resource):
            continue
        subject = evaluation.get("subject")
        if not isinstance(subject, dict):
            subject = {}
            evaluation["subject"] = subject
        subject["type"] = subject_type
        if subject_id:
            subject["id"] = subject_id
        else:
            subject.pop("id", None)
