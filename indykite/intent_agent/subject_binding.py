"""Schema-derived validation of the LLM's output (TDD 4.8) and deterministic subject binding (TDD 4.7)."""

import json
import re
from dataclasses import dataclass
from typing import Any

from indykite.intent_agent.cel import filter_errors
from indykite.intent_agent.errors import (
    SCHEMA_BLANK_SUFFIX,
    SCHEMA_MEMBERSHIP_SUFFIX,
    IntentValidationError,
    SubjectTypeMismatchError,
    no_intent_error,
)


@dataclass(frozen=True, slots=True)
class RequestSubject:
    """The caller-supplied subject identity for one request resolved and sent by
    IAG itself, not derived from the Bearer token, which now authenticates the
    calling client rather than any subject. `id` may be blank (an anonymous caller
    a Subject Search is the LLM's per-evaluation marker instead); `type` may not.
    """

    id: str | None
    type: str


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """This request's fetched AuthZEN policies used to accept a policy-derived
    action.name when the evaluation is marked action_derivation_rule="policy_based", see
    _action_name_error - and whether the fetch itself failed after retries, which
    normalize_intent_response tags onto every evaluation's context.parameters (see
    _set_policy_fetch_failed) so the receiving agent knows this answer used the data schema
    only. Bundled together since both come from one per-request PolicyClient.get_cached() call
    and normalize_intent_response needs both; separately they would push its own argument count
    over the lint limit.
    """

    policies: list[dict[str, Any]] | None
    fetch_failed: bool


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


# SKILL.md's marker for "action.name came from a policy, not the schema's edge relations" -
# same context.parameters convention as subject_source, a different key so the two markers
# never collide on one evaluation.
_ACTION_DERIVATION_KEY = "action_derivation_rule"
_ACTION_DERIVATION_POLICY = "policy_based"
_POLICY_FETCH_FAILED_KEY = "policy_fetch_failed"


def _policy_actions_for_resource(policies: list[dict[str, Any]] | None, resource_type: str) -> set[str]:
    """Every action name a fetched policy defines for evaluations against this resource type -
    from policy.actions where policy.resource.type matches. Confirmed against a real captured
    GET /access/v1/policies response (ENG-9671):
    {"results": [{"policy": {"actions": [...], "resource": {"type": ...}, "subject": {...},
    "condition": {...}}, "tags": [...]}]} - subject.type is not checked here too, since the
    fetch itself is already scoped to the caller's subject_type (PolicyClient.get_cached).
    """
    if not policies:
        return set()
    names: set[str] = set()
    for entry in policies:
        policy = entry.get("policy") if isinstance(entry, dict) else None
        if not isinstance(policy, dict):
            continue
        resource = policy.get("resource")
        if not isinstance(resource, dict) or resource.get("type") != resource_type:
            continue
        actions = policy.get("actions")
        if isinstance(actions, list):
            names.update(name for name in actions if isinstance(name, str) and name)
    return names


def _action_derivation_rule(evaluation: dict[str, Any]) -> object:
    """Return the evaluation's context.parameters.action_derivation_rule marker, mirroring
    _subject_source - a malformed context/parameters reads as no marker at all.
    """
    context = evaluation.get("context")
    if not isinstance(context, dict):
        return None
    parameters = context.get("parameters")
    if not isinstance(parameters, dict):
        return None
    return parameters.get(_ACTION_DERIVATION_KEY)


def _action_name_error(
    action: dict[str, Any],
    resource_type: object,
    edge_relations: set[str],
    policies: list[dict[str, Any]] | None,
    marker: object,
) -> str | None:
    """Describe why action.name isn't acceptable, or None if it is. `action` is already known
    to be an object (see _evaluation_errors); `marker` is the evaluation's own
    context.parameters.action_derivation_rule value (_action_derivation_rule).

    Valid two ways: as a schema edge relation, always (marked or not - unchanged from before
    ENG-9671), or as a policy-derived action for this evaluation's resource.type, but only when
    marker == "policy_based". Unmarked, a policy-only action name is rejected the same as any
    other off-schema value - the retry loop (validation_feedback_error) is what pushes the LLM
    to add the marker, not a silent acceptance of an unmarked claim.
    """
    name = action.get("name")
    reason = _vocabulary_reason("action.name", name, edge_relations)
    if reason is None or _blank(name) or not isinstance(resource_type, str):
        return reason
    if name in _policy_actions_for_resource(policies, resource_type) and marker == _ACTION_DERIVATION_POLICY:
        return None
    return reason


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
    errors = [
        f"context.{key} is not {_TYPE_NAMES[expected]}"
        for key, expected in _CONTEXT_KEY_TYPES.items()
        if (value := context.get(key)) is not None and not isinstance(value, expected)
    ]
    filters = context.get("filters")
    if not _blank(filters):
        # A blank (or non-string, reported above) filter is not checked as CEL - a blank one
        # means "no filter" and is dropped by _normalize_contexts.
        errors.extend(filter_errors(filters))
    return errors


def _evaluation_errors(
    evaluation: object,
    node_types: set[str],
    edge_relations: set[str],
    policies: list[dict[str, Any]] | None,
) -> list[str]:
    """Return every reason this evaluation can't be accepted, empty if it can.

    Must run before _bind_subject, which overwrites subject.type/id regardless of what the
    LLM produced on every evaluation but the marked exceptions - and those keep a subject.type
    only this check has vetted. A key entirely absent (e.g. no `subject` for Access
    Evaluation, no `action` for Action Search - see SKILL.md) is a legitimate
    per-evaluation-kind omission, not a failure; a key present but malformed (wrong
    type, or a blank/off-schema type/name inside it) means the LLM tried and failed to
    correlate it. subject is exempt from the malformed-shape check (though not the
    blank/off-schema one) since _bind_subject replaces a non-object subject outright either
    way. context has no schema vocabulary to check against, only a shape (_context_errors).
    action.name is additionally valid as a policy-derived action for resource.type when marked
    action_derivation_rule="policy_based" (ENG-9671) - see _action_name_error.
    """
    if not isinstance(evaluation, dict):
        return ["evaluation is not an object"]
    errors: list[str] = []
    resource = evaluation.get("resource")
    resource_type = resource.get("type") if isinstance(resource, dict) else None
    if not isinstance(resource, dict):
        errors.append("resource is missing or not an object")
    else:
        errors.append(_vocabulary_reason("resource.type", resource_type, node_types))
    action = evaluation.get("action")
    if action is not None:
        if not isinstance(action, dict):
            errors.append("action is not an object")
        else:
            marker = _action_derivation_rule(evaluation)
            errors.append(_action_name_error(action, resource_type, edge_relations, policies, marker))
    subject = evaluation.get("subject")
    if isinstance(subject, dict):
        errors.append(_vocabulary_reason("subject.type", subject.get("type"), node_types))
    errors.extend(_context_errors(evaluation.get("context")))
    return [error for error in errors if error]


def validate_evaluations(
    schema: dict[str, Any],
    evaluations: list[object],
    policies: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Check every evaluation's subject.type / action.name / resource.type against the
    schema's node types and edge relations (TDD 4.8 post-assembly check), and its context
    against the PRD's parameters/filters/qualifiers shape. Returns one message per problem,
    prefixed with the evaluation's index so the LLM can tell which entry to fix (they
    become the `reason` lines of errors.validation_feedback_error); empty means the list
    is acceptable. `policies` (ENG-9671) additionally allows a marked action.name to come from
    a policy instead of the schema - see _action_name_error.
    """
    node_types = _node_types(schema)
    edge_relations = _edge_relations(schema)
    return [
        f"evaluations[{index}]: {error}"
        for index, evaluation in enumerate(evaluations)
        for error in _evaluation_errors(evaluation, node_types, edge_relations, policies)
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
    subject: RequestSubject,
    policy_context: PolicyContext,
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
    errors = validate_evaluations(schema, evaluations, policy_context.policies)
    if errors:
        raise IntentValidationError(errors)

    _bind_subject(data, subject_id=subject.id, subject_type=subject.type)
    _normalize_contexts(data)
    _set_policy_fetch_failed(data, fetch_failed=policy_context.fetch_failed)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _set_policy_fetch_failed(data: dict[str, Any], *, fetch_failed: bool) -> None:
    """Set every evaluation's context.parameters.policy_fetch_failed to reflect fetch_failed -
    a system-set flag, never the LLM's to produce (ENG-9671, review). context.parameters is an
    open object in the structured output, so without this an unmarked/hallucinated
    policy_fetch_failed the model puts there on a successful-fetch response would pass through
    untouched, contradicting policy_context/the audit record. On success (fetch_failed=False),
    any LLM-supplied value for this reserved key is stripped rather than trusted; on failure,
    every evaluation is unconditionally marked True regardless of what the model produced.
    Runs after _normalize_contexts, so context is already settled into an object on every
    evaluation.
    """
    for evaluation in data.get("evaluations", []):
        if not isinstance(evaluation, dict):
            continue
        context = evaluation["context"]
        parameters = context.get("parameters")
        if fetch_failed:
            if not isinstance(parameters, dict):
                parameters = {}
                context["parameters"] = parameters
            parameters[_POLICY_FETCH_FAILED_KEY] = True
        elif isinstance(parameters, dict):
            parameters.pop(_POLICY_FETCH_FAILED_KEY, None)


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


# SKILL.md's marker for "this evaluation's subject is not the caller" - set in
# context.parameters, the one part of an evaluation already documented as free-form
# prompt-derived data. One key with two values, so an evaluation can never claim both:
# "prompt" means a second subject the LLM identified in the prompt (trusted only behind a
# gate, see _subject_from_prompt), "search" means a Subject Search whose subject.type is the
# entity type searched for (see _subject_search). Its absence, or any other value, is the
# default and safe fallback: override, exactly as when the evaluation is about the caller.
_SUBJECT_SOURCE_KEY = "subject_source"
_SUBJECT_SOURCE_PROMPT = "prompt"
_SUBJECT_SOURCE_SEARCH = "search"


def _subject_source(evaluation: dict[str, Any]) -> object:
    """Return the evaluation's context.parameters.subject_source marker, or None when the
    context or parameters aren't objects - a malformed marker reads as no marker at all.
    """
    context = evaluation.get("context")
    if not isinstance(context, dict):
        return None
    parameters = context.get("parameters")
    if not isinstance(parameters, dict):
        return None
    return parameters.get(_SUBJECT_SOURCE_KEY)


def _subject_search(evaluation: dict[str, Any]) -> bool:
    """True if the LLM marked this evaluation as a Subject Search ("who can do Y on Z?", see
    SKILL.md) and gave it a subject.type to search for - the other case _bind_subject must not
    override, since that type is a search parameter (the entity type being looked for), not
    anyone's identity. A marker without a usable subject.type falls back to the safe default
    (override), same as a malformed marker.
    """
    if _subject_source(evaluation) != _SUBJECT_SOURCE_SEARCH:
        return False
    subject = evaluation.get("subject")
    return isinstance(subject, dict) and not _blank(subject.get("type"))


def _gate_resource(data: dict[str, Any]) -> tuple[str, str] | None:
    """Return (type, id) of evaluations[0]'s resource - the caller-anchored gate evaluation the
    second-subject pattern hinges on (SKILL.md: shaped like Access Evaluation, "resource is the
    other subject itself") - or None if evaluations[0] isn't shaped that way. The marker alone
    (_subject_from_prompt) is never enough to trust a claimed subject; it must also match a real
    gate: a specific action against a specific, identified resource - not a resource-search-shaped
    entry (blank id) or an action-search-shaped one (no action at all), neither of which checks
    anything specific enough to gate on. Nor is a Subject Search (_subject_search): it is not
    bound to the caller, so it checks nothing about the caller at all - without this a search at
    evaluations[0] would unlock a marked evaluations[1] and no evaluation would name the caller.
    """
    evaluations = data.get("evaluations", [])
    if not evaluations or not isinstance(evaluations[0], dict):
        return None
    gate = evaluations[0]
    if _subject_search(gate):
        return None
    action = gate.get("action")
    if not isinstance(action, dict) or _blank(action.get("name")):
        return None
    resource = gate.get("resource")
    if not isinstance(resource, dict):
        return None
    resource_type, resource_id = resource.get("type"), resource.get("id")
    if _blank(resource_type) or _blank(resource_id):
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
    if _subject_source(evaluation) != _SUBJECT_SOURCE_PROMPT:
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
    surface where the LLM could otherwise claim an identity for the subject itself.

    Two exceptions, both opted into by the LLM through context.parameters.subject_source:

    - "prompt": an evaluation *after the first* about a second subject the LLM identified in
      the prompt, whose subject matches evaluations[0]'s gate resource (_subject_from_prompt),
      keeps the subject it produced. evaluations[0] is deliberately *never* eligible, marked
      or not: it is the caller-anchored evaluation the whole pattern hinges on, so a response
      with only one evaluation (no second subject at all) must always come back bound to the
      caller, and can never be trusted on the LLM's say-so alone.
    - "search": a Subject Search evaluation (_subject_search), at any index, keeps the
      subject.type the LLM chose and never carries an id - the search discovers subjects, so
      neither the caller's id nor whatever the LLM put there is bound into it.

    Every other evaluation - the normal single-subject case, any evaluation the LLM didn't mark
    or marked for a subject the gate doesn't cover - gets the override regardless of what the
    LLM put there, same as before.
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
        if _subject_search(evaluation):
            subject.pop("id", None)
            continue
        subject["type"] = subject_type
        if subject_id:
            subject["id"] = subject_id
        else:
            subject.pop("id", None)
