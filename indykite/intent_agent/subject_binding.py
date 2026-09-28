"""Schema-derived validation of the LLM's output (TDD 4.8) and deterministic subject binding (TDD 4.7)."""

import json
import re
from dataclasses import dataclass
from typing import Any

from indykite.intent_agent.cel import filter_errors
from indykite.intent_agent.errors import (
    SCHEMA_BLANK_SUFFIX,
    SCHEMA_MEMBERSHIP_SUFFIX,
    SCHEMA_RELATIONSHIP_SUFFIX,
    IntentValidationError,
    SubjectTypeMismatchError,
    no_intent_error,
)
from indykite.intent_agent.input_schema import digital_twin_types


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


def _node_properties(schema: dict[str, Any]) -> dict[str, set[str]]:
    """Each node type's own property names, as the data schema observed them.

    A type whose properties are missing or malformed maps to an empty set, which disables the
    filter-property check for it (cel.filter_errors) rather than rejecting every filter on it.
    """
    nodes = schema.get("graph", {}).get("nodes", {})
    if not isinstance(nodes, dict):
        return {}
    properties: dict[str, set[str]] = {}
    for node_type, node in nodes.items():
        metadata = node.get("metadata") if isinstance(node, dict) else None
        declared = metadata.get("properties") if isinstance(metadata, dict) else None
        names = {name for name in declared if isinstance(name, str)} if isinstance(declared, dict) else set()
        properties[node_type] = names
    return properties


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

    Valid three ways: as a schema edge relation, always (marked or not - unchanged from before
    ENG-9671); as a policy-derived action for this evaluation's resource.type, but only when
    marker == "policy_based"; or, when the schema has no edges at all *and* no policies were
    fetched, unconstrained - the pre-ENG-9671 behavior for a schema with nodes but no edges yet
    (see _vocabulary_reason). Unmarked, or against the wrong resource.type, a policy-only action
    name is rejected the same as any other off-schema value - the retry loop
    (validation_feedback_error) is what pushes the LLM to add the marker, not a silent
    acceptance of an unmarked claim.

    This checks schema and policy membership directly rather than delegating to
    _vocabulary_reason: that helper's own "empty allowed set" carve-out (for a schema with no
    edges yet) returns None for any non-blank name regardless of policies, which would silently
    skip the policy-membership/marker check below whenever the schema simply has no edges -
    letting an arbitrary action, or a policy action against the wrong resource.type, through
    unmarked. The unconstrained carve-out here only applies when there is truly no vocabulary
    (schema or policy) to validate against at all.
    """
    name = action.get("name")
    if _blank(name):
        # Only reachable through a direct validate_evaluations call: normalize_intent_response
        # drops blank-named evaluations first (_drop_unnamed_actions) instead of retrying them.
        return f"action.name {SCHEMA_BLANK_SUFFIX}"
    if name in edge_relations:
        return None
    if (
        isinstance(resource_type, str)
        and marker == _ACTION_DERIVATION_POLICY
        and name in _policy_actions_for_resource(policies, resource_type)
    ):
        return None
    if not edge_relations and not policies:
        return None
    return f"action.name {name!r} {SCHEMA_MEMBERSHIP_SUFFIX}"


# The PRD's context sub-shapes (TDD 3.2) and what each must be when present - the same three
# keys input_schema._Context declares for the structured output.
_CONTEXT_KEY_TYPES: dict[str, type] = {"filters": str, "parameters": dict, "qualifiers": dict}
_TYPE_NAMES = {str: "a string", dict: "an object"}


def _context_errors(context: object, properties: set[str] | None = None) -> list[str]:
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
        # means "no filter" and is dropped by _normalize_contexts. `properties` are the resource
        # type's own, so a filter reading a property that type doesn't have is reported too.
        errors.extend(filter_errors(filters, properties))
    return errors


def _evaluation_errors(
    evaluation: object,
    node_types: set[str],
    edge_relations: set[str],
    policies: list[dict[str, Any]] | None,
    node_properties: dict[str, set[str]] | None = None,
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
    way. context is checked for its shape and, where the schema observed properties for
    resource.type, for filters reading a property that type doesn't have (_context_errors).
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
        if resource.get("id") is not None and not isinstance(resource["id"], str):
            errors.append("resource.id is not a string")
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
    if _subject_source(evaluation) == _SUBJECT_SOURCE_SEARCH and not isinstance(subject, dict):
        # Seen live (flash-lite): the marker is set but the `subject` object is left out, mixing up
        # SKILL.md's "omit subject" rule for the other evaluation kinds with the Subject Search
        # exception. Without this the marker is ignored, _bind_subject binds the caller, and the
        # caller then fails the relationship check - which sends the retry after the action name
        # instead of the missing type, and it answers with an empty array instead. A subject that
        # is present but blank already reports itself above; this is the case nothing else sees.
        errors.append(
            'context.parameters.subject_source is "search" but subject.type is missing - a Subject '
            "Search must name the subject type to search for"
        )
    properties = (node_properties or {}).get(resource_type) if isinstance(resource_type, str) else None
    errors.extend(_context_errors(evaluation.get("context"), properties))
    return [error for error in errors if error]


def _edge_triples(schema: dict[str, Any]) -> set[tuple[str, str, str]]:
    """Every well-formed (source, relation, target) edge of the schema."""
    edges = schema.get("graph", {}).get("edges", [])
    if not isinstance(edges, list):
        return set()
    return {
        (edge["source"], edge["relation"], edge["target"])
        for edge in edges
        if isinstance(edge, dict) and all(not _blank(edge.get(key)) for key in ("source", "relation", "target"))
    }


def _relationship_reason(
    evaluation: dict[str, Any],
    *,
    caller_type: str,
    previous_resource_type: str | None,
    triples: set[tuple[str, str, str]],
    policies: list[dict[str, Any]] | None,
) -> str | None:
    """Why the caller can't hold this evaluation's relationship, or None if they can. The action
    must be an edge from the caller's own type to the resource type, or an action a fetched
    policy grants for that resource type (the listing is already scoped to the caller's subject
    type - PolicyClient.get_cached; permission labels such as CAN_ACCESS are granted there, not
    by whichever pair of node types happens to hold such an edge in the graph) - or, on a
    dependent hop of a chain, an edge from the previous hop's resource type, `previous_resource_type`
    being None unless that previous hop was itself grounded by this check (the dependency marker
    is the model's, so it can't vouch for a hop on its own). Seen live otherwise: "what accounts
    do I own?" in a graph where nothing relates a User to an Account answered with `REGARDING`,
    a `Ticket -> Account` edge, a forced match that must not reach the caller.
    """
    action, resource = evaluation.get("action"), evaluation.get("resource")
    name = action.get("name") if isinstance(action, dict) else None
    resource_type = resource.get("type") if isinstance(resource, dict) else None
    if _blank(name) or _blank(resource_type):
        return None
    if (caller_type, name, resource_type) in triples or name in _policy_actions_for_resource(policies, resource_type):
        return None
    dependent = _subject_source_parameters(evaluation).get("depends_on_previous_evaluations") is True
    if dependent and previous_resource_type is not None and (previous_resource_type, name, resource_type) in triples:
        return None
    return (
        f"action.name {name!r} for subject type {caller_type!r} on resource type {resource_type!r} "
        f"{SCHEMA_RELATIONSHIP_SUFFIX}"
    )


def _is_caller(subject: dict[str, Any], caller_type: str, caller_id: str | None) -> bool:
    """True when `subject` names the caller's own identity - type and id both."""
    return subject.get("type") == caller_type and not _blank(caller_id) and subject.get("id") == caller_id


def _caller_relationship_errors(
    schema: dict[str, Any],
    evaluations: list[object],
    policies: list[dict[str, Any]] | None,
    caller_type: str,
    caller_id: str | None,
) -> list[str]:
    """The relationship check for every evaluation that carries a subject: the caller's own
    (bound, or a "second subject" that is in fact the caller's identity - the marker never
    exempts the caller from their own check) against the caller's type, a trusted second
    subject against its own type. A marked Subject Search discovers subjects and is left to the
    vocabulary check alone. A second subject additionally passes with an action marked
    `policy_based`: the golden answers (adversarial_subject_spoof_test_2) name a permission
    about a third party that neither an edge nor a fetched policy - scoped to the caller's type -
    grounds, so that claim is IBAC's to decide; an *unmarked* relationship label must be an edge
    from the second subject's own type. A hop only counts as the "previous hop" for the next
    evaluation's dependent-hop rule when it passed this check and named an action.
    """
    triples = _edge_triples(schema)
    gate = _gate_resource({"evaluations": evaluations})
    errors: list[str] = []
    previous_resource_type: str | None = None
    for index, evaluation in enumerate(evaluations):
        if not isinstance(evaluation, dict) or _subject_search(evaluation):
            previous_resource_type = None
            continue
        second_subject = (
            index > 0
            and _subject_from_prompt(evaluation, gate)
            and not _is_caller(evaluation["subject"], caller_type, caller_id)
        )
        # The fetched policies are the caller's subject type's: they ground the caller's own
        # evaluations only. A second subject passes on its own edge, a grounded dependent hop, or
        # the explicit policy_based claim below - never on the caller's policies.
        reason = _relationship_reason(
            evaluation,
            caller_type=evaluation["subject"]["type"] if second_subject else caller_type,
            previous_resource_type=previous_resource_type,
            triples=triples,
            policies=None if second_subject else policies,
        )
        if reason and second_subject and _action_derivation_rule(evaluation) == _ACTION_DERIVATION_POLICY:
            reason = None
        if reason:
            errors.append(f"evaluations[{index}]: {reason}")
        # Only a hop that named an action and passed the check establishes a relationship the
        # next hop may build on - an Action Search (no action) checks nothing and grounds nothing.
        action, resource = evaluation.get("action"), evaluation.get("resource")
        named = isinstance(action, dict) and not _blank(action.get("name"))
        grounded = reason is None and named and isinstance(resource, dict)
        previous_resource_type = resource.get("type") if grounded else None
    return errors


def validate_evaluations(
    schema: dict[str, Any],
    evaluations: list[object],
    policies: list[dict[str, Any]] | None = None,
    *,
    caller_type: str | None = None,
    caller_id: str | None = None,
) -> list[str]:
    """Check every evaluation's subject.type / action.name / resource.type against the
    schema's node types and edge relations (TDD 4.8 post-assembly check), and its context
    against the PRD's parameters/filters/qualifiers shape. Returns one message per problem,
    prefixed with the evaluation's index so the LLM can tell which entry to fix (they
    become the `reason` lines of errors.validation_feedback_error); empty means the list
    is acceptable. `policies` (ENG-9671) additionally allows a marked action.name to come from
    a policy instead of the schema - see _action_name_error. With `caller_type`, every
    evaluation bound to the caller must also name a relationship the caller's type can hold
    (_relationship_reason) - checked only once the vocabulary itself is clean, and only when
    there is a relationship vocabulary to check against: a schema with no edges and no policies
    fetched leaves action.name unconstrained (_action_name_error's carve-out), and a check that
    then failed every caller-bound action would turn that legitimate state into a refusal.
    """
    node_types = _node_types(schema)
    edge_relations = _edge_relations(schema)
    node_properties = _node_properties(schema)
    errors = [
        f"evaluations[{index}]: {error}"
        for index, evaluation in enumerate(evaluations)
        for error in _evaluation_errors(evaluation, node_types, edge_relations, policies, node_properties)
    ]
    if caller_type is not None and not errors and (edge_relations or policies):
        errors = _caller_relationship_errors(schema, evaluations, policies, caller_type, caller_id)
    return errors


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


def looks_like_evaluations(llm_text: str) -> bool:
    """True when `llm_text` is an answer normalize_intent_response would treat as an extraction:
    a JSON object with an `evaluations` key, or a bare JSON array of evaluations, either one
    optionally inside a ```json fence - exactly the leniency _parse_llm_json applies, so an
    answer given without activating the skill can never slip past agent._extract_with_retries
    as "free text" in a shape this function would have accepted.
    """
    try:
        data = _parse_llm_json(llm_text)
    except IntentValidationError:
        return False
    return isinstance(data, list) or (isinstance(data, dict) and "evaluations" in data)


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
    _drop_unnamed_actions(data)
    evaluations = _drop_off_twin_subject_searches(data, schema)
    if not evaluations:
        return json.dumps(no_intent_error(original_prompt), indent=2, ensure_ascii=False)
    # Settle resource ids first: the gate check below reads evaluations[0].resource.id, and a
    # gate the model wrote in the filter form must be recognised as the same identified resource.
    _drop_blank_resource_ids(data)
    _lift_external_id_filters(data)
    _complete_second_subject_types(data)
    _bind_caller_resources(data, subject_id=subject.id, subject_type=subject.type)
    _mark_policy_derived_actions(data, schema, policy_context.policies, caller_type=subject.type)
    errors = validate_evaluations(
        schema, evaluations, policy_context.policies, caller_type=subject.type, caller_id=subject.id
    )
    if errors:
        raise IntentValidationError(errors)

    _bind_subject(data, subject_id=subject.id, subject_type=subject.type)
    _normalize_contexts(data)
    _drop_duplicate_evaluations(data)
    _set_policy_fetch_failed(data, fetch_failed=policy_context.fetch_failed)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _mark_policy_derived_actions(
    data: dict[str, Any],
    schema: dict[str, Any],
    policies: list[dict[str, Any]] | None,
    *,
    caller_type: str,
) -> None:
    """Set `action_derivation_rule: "policy_based"` where the action is policy-derived by
    definition, so the marker is a statement of fact the system makes itself rather than a claim
    it needs the model to attest to. Two cases: an action a fetched policy grants for its resource
    type that is not the subject's own graph edge to that type (an edge label reused elsewhere in
    the graph doesn't make it a relationship the subject holds), and a dependent hop of a chain
    (`depends_on_previous_evaluations: true`) whose action is not an edge from the evaluation's
    own subject type - SKILL.md's rule, and what every golden dependent hop carries.
    validate_evaluations still rejects an unmarked policy action when called directly, and still
    rejects any name that is neither an edge nor a policy action for that resource type. Seen
    live (flash-lite): the model picks the right action and forgets the marker, and the retry
    that followed turned a correct chain into a refusal.
    """
    triples = _edge_triples(schema)
    gate = _gate_resource(data)
    for index, evaluation in enumerate(data.get("evaluations", [])):
        if not isinstance(evaluation, dict):
            continue
        action, resource = evaluation.get("action"), evaluation.get("resource")
        name = action.get("name") if isinstance(action, dict) else None
        resource_type = resource.get("type") if isinstance(resource, dict) else None
        if _blank(name) or _blank(resource_type):
            continue
        parameters = _subject_source_parameters(evaluation)
        # The subject type the evaluation will actually carry: its own only where _bind_subject
        # keeps it (a Subject Search, a trusted second subject), the caller's everywhere else -
        # a schema-valid but wrong type the model wrote on a caller-bound evaluation is overwritten
        # and must not decide the marker.
        second_subject = index > 0 and _subject_from_prompt(evaluation, gate)
        keeps_own_subject = _subject_search(evaluation) or second_subject
        own_type = evaluation["subject"]["type"] if keeps_own_subject else caller_type
        # The fetched policies are the caller's subject type's, so they can only speak for an
        # evaluation that ends up carrying that type. Marking a second subject of a *different*
        # type would hand _caller_relationship_errors the policy_based claim it clears its reason
        # on - the claim it deliberately refuses to read off the caller's policies - and so ground
        # a relationship nothing established (a User policy can't say what a Customer may do).
        # Left unmarked, such an evaluation stands or falls on the model's own explicit claim,
        # which is what a second subject is meant to rest on. A Subject Search is unaffected: its
        # subject is unidentified and _caller_relationship_errors never grounds it at all.
        if second_subject and own_type != caller_type:
            continue
        # Policy-derived unless the action is the subject's own graph edge to that resource type -
        # judged on the (policy, resource type) pair, not on whether the label also appears on some
        # unrelated edge elsewhere in the graph (CAN_ACCESS usually does).
        policy_granted = name in _policy_actions_for_resource(policies, resource_type)
        policy_only = policy_granted and (own_type, name, resource_type) not in triples
        dependent_off_type = (
            parameters.get("depends_on_previous_evaluations") is True and (own_type, name, resource_type) not in triples
        )
        if not (policy_only or dependent_off_type):
            continue
        # A missing context / parameters is created; a malformed one (a string, a list) is left
        # alone so validate_evaluations still reports the shape error.
        context = evaluation.get("context")
        if context is None:
            context = {}
            evaluation["context"] = context
        elif not isinstance(context, dict):
            continue
        parameters = context.get("parameters")
        if parameters is None:
            parameters = {}
            context["parameters"] = parameters
        elif not isinstance(parameters, dict):
            continue
        parameters[_ACTION_DERIVATION_KEY] = _ACTION_DERIVATION_POLICY


def _drop_unnamed_actions(data: dict[str, Any]) -> list[object]:
    """Drop every evaluation whose `action` object carries a blank string `name` and return
    what is left (also written back to `data`). A present-but-blank action is the model's "no
    action in the vocabulary fits this part of the prompt" - SKILL.md tells it to leave such a
    part out, and treating the blank as a validation error to retry was seen live (flash-lite)
    to push it into forcing a match on an unrelated edge on the second attempt instead. An
    absent `action` key is different: that is the Action Search shape and stays. A null name or
    an empty `action` object is neither - malformed output, left for validation to report and
    the retry to mend, as before.
    """
    evaluations = data.get("evaluations", [])
    kept = [
        evaluation
        for evaluation in evaluations
        if not (
            isinstance(evaluation, dict)
            and isinstance(evaluation.get("action"), dict)
            and isinstance(evaluation["action"].get("name"), str)
            and not evaluation["action"]["name"].strip()
        )
    ]
    data["evaluations"] = kept
    return kept


def _drop_off_twin_subject_searches(data: dict[str, Any], schema: dict[str, Any]) -> list[object]:
    """Drop every Subject Search whose searched subject.type is a node type of the schema that
    isn't marked DigitalTwin, and return what is left (also written back to `data`). Only
    DigitalTwin nodes can be subjects (SKILL.md: the candidate types of a "who can" question are
    exactly the DigitalTwin types), so a search over a resource-only type asks for subjects of a
    kind that can't be one - not a different answer, no answer. Seen live (flash-lite): "who can
    access the document?" in a model whose one DigitalTwin type is User came back with a second,
    `Customer` search next to the right one. A type the schema doesn't define at all is left for
    validate_evaluations to feed back (the model may still correct it on the retry), and a
    schema that marks no DigitalTwin type at all is left alone: there is nothing to judge the
    candidates against. An evaluation bound to the caller isn't a candidate and is never touched
    here - _bind_subject overwrites its subject anyway. The caller's own type is judged like any
    other: a search over it when it isn't a DigitalTwin (or over `_Application`, which never is
    a candidate) is dropped too, and a response left empty by this is the no-intent error.
    """
    evaluations = data.get("evaluations", [])
    twins = set(digital_twin_types(schema))
    if not twins:
        return evaluations
    resource_only = _node_types(schema) - twins
    kept = [
        evaluation
        for evaluation in evaluations
        if not (
            isinstance(evaluation, dict)
            and _subject_search(evaluation)
            and evaluation["subject"]["type"] in resource_only
        )
    ]
    data["evaluations"] = kept
    return kept


def _drop_blank_resource_ids(data: dict[str, Any]) -> None:
    """Remove a blank `resource.id`: the structured output requires the field (input_schema
    _Resource - a required field is filled far more reliably than an optional one), so "many
    resources" arrives as the empty string, and the caller must see the AuthZEN shape for a
    search - no id key - not an empty one.
    """
    for evaluation in data.get("evaluations", []):
        if not isinstance(evaluation, dict):
            continue
        resource = evaluation.get("resource")
        if not isinstance(resource, dict) or "id" not in resource:
            continue
        resource_id = resource["id"]
        # Only a null or blank-string id is "no id"; a non-string id is left for validation
        # to report (_evaluation_errors), not silently turned into a search over every resource.
        if resource_id is None or (isinstance(resource_id, str) and not resource_id.strip()):
            del resource["id"]


# One `resource.<property> == '<literal>'` clause, either way round, single or double quotes.
_EQUALITY_CLAUSE = re.compile(
    r"""^\s*(?:resource\.(?P<prop>\w+)\s*==\s*(?P<q1>['"])(?P<a>[^'"]*)(?P=q1)"""
    r"""|(?P<q2>['"])(?P<b>[^'"]*)(?P=q2)\s*==\s*resource\.(?P<prop2>\w+))\s*$"""
)
# The properties that name one resource instance: the platform's external_id *is* the id
# (SKILL.md), and the model sometimes writes resource.id for it directly.
_ID_PROPERTIES = frozenset({"external_id", "id"})
# `resource["external_id"]` selects the same property as `resource.external_id` in CEL, so the
# bracket spelling is folded into the dot one before the clauses are read - otherwise the same
# filter would lift or not depending on how the model happened to write it.
_RESOURCE_INDEX = re.compile(r"""resource\[\s*(?P<q>['"])(?P<prop>\w+)(?P=q)\s*\]""")


def _filter_identifier(filters: str) -> str | None:
    """The single resource id `filters` pins, or None when it says anything else.

    Two forms qualify. A lone `resource.external_id == 'x'`, and a chain of `||` clauses that all
    compare to the *same* literal where one of them is the id: seen live, flash-lite hedges with
    `resource.external_id == 'Volvo' || resource.email == 'Volvo'` when it is unsure which
    property carries the name. Every alternative there names the instance the id already names,
    so collapsing them into resource.id keeps the question and drops only the hedge. A `&&`, a
    parenthesised group, or a clause comparing to a different literal is a real condition on
    which resources qualify, and is left alone.
    """
    filters = _RESOURCE_INDEX.sub(lambda match: f"resource.{match.group('prop')}", filters)
    if "&&" in filters or "(" in filters or ")" in filters:
        return None
    properties: set[str] = set()
    literals: set[str] = set()
    for clause in filters.split("||"):
        match = _EQUALITY_CLAUSE.match(clause)
        if match is None:
            return None
        properties.add(match.group("prop") or match.group("prop2"))
        first = match.group("a")
        literals.add(first if first is not None else match.group("b"))
    if not properties & _ID_PROPERTIES or len(literals) != 1:
        return None
    identifier = literals.pop()
    return None if _blank(identifier) else identifier


def _lift_external_id_filters(data: dict[str, Any]) -> None:
    """Move a filter that only identifies one resource (_filter_identifier) into `resource.id`
    when the model left the id blank. The platform's external_id *is* the resource id (SKILL.md:
    id = the node's external_id), so the two forms name the same single resource - but only the
    id form is the AuthZEN shape for one named instance, it's what the golden answers expect, and
    an Access Evaluation without it is rejected downstream ("missing field resource.id"). Seen
    live with flash-lite, which prefers the filter form despite being told otherwise; this makes
    the response independent of that habit without guessing at any other filter's meaning.

    Runs before validate_evaluations on purpose: a hedged filter is consumed here, so the
    invented property it hedged with (`resource.email` on a Car) never reaches the property check
    as a retry-worthy error. What the lift can't resolve, that check still reports.
    """
    for evaluation in data.get("evaluations", []):
        if not isinstance(evaluation, dict):
            continue
        resource, context = evaluation.get("resource"), evaluation.get("context")
        if not isinstance(resource, dict) or not isinstance(context, dict):
            continue
        resource_id = resource.get("id")
        # Only an absent, null or blank-string id is "no id" to fill; any other type is left as
        # written for validation to report (resource.id is not a string), not overwritten.
        if resource_id is not None and (not isinstance(resource_id, str) or resource_id.strip()):
            continue
        filters = context.get("filters")
        if not isinstance(filters, str):
            continue
        # None for anything that isn't one identified resource, including
        # `resource.external_id == '  '`, which names nothing and stays the filter it is.
        lifted = _filter_identifier(filters)
        if lifted is None:
            continue
        resource["id"] = lifted
        del context["filters"]


def _drop_duplicate_evaluations(data: dict[str, Any]) -> None:
    """Keep the first of any evaluations that are identical once bound and normalised. A
    repeated evaluation asks the exact same question twice, so it carries nothing for the
    caller; the model (flash-lite, seen live) sometimes emits the same entry two times over.
    Only literal duplicates go - two evaluations differing in any key are both kept, in order.
    """
    evaluations = data.get("evaluations", [])
    seen: list[object] = []
    kept: list[object] = []
    for index, evaluation in enumerate(evaluations):
        following = evaluations[index + 1] if index + 1 < len(evaluations) else None
        # A repeat that a dependent hop builds on stays: dropping it would make the hop read as
        # depending on whatever precedes it instead, a chain the relationship check never saw.
        anchors_a_dependent_hop = (
            isinstance(following, dict)
            and _subject_source_parameters(following).get("depends_on_previous_evaluations") is True
        )
        if evaluation in seen and not anchors_a_dependent_hop:
            continue
        seen.append(evaluation)
        kept.append(evaluation)
    data["evaluations"] = kept


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
    """Settle every evaluation's context into the PRD shape after validation and binding:
    always an object (`{}` when the LLM gave none), with the structured output's nullability
    artefacts removed - a null key, or a blank `filters` string, both mean "not applicable",
    and the caller shouldn't have to tell those apart from an absent key - and with the
    `subject_source` / `resource_source` markers removed: they are instructions from the model
    to the binding, not prompt-derived data, and once binding has consumed them the caller has
    no use for them (the golden fixtures carry no such keys). A `parameters` object left empty
    by that removal goes too. Everything else is passed through verbatim: context is
    user-controlled data, and it's the downstream consumer's job to treat it as untrusted
    (TDD 9), not this agent's to rewrite it.
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
        parameters = context.get("parameters")
        if isinstance(parameters, dict) and not _MARKER_KEYS.isdisjoint(parameters):
            parameters = {key: value for key, value in parameters.items() if key not in _MARKER_KEYS}
            if parameters:
                context["parameters"] = parameters
            else:
                del context["parameters"]
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
# The resource-side counterpart: "this evaluation's resource is the caller themself" ("who can
# access me?"). The model can't know the caller's identity, so it marks the evaluation and
# _bind_caller_resources fills the caller's type/id in - only ever the caller's own identity, so
# a prompt can't use it to point the evaluation at anyone else.
_RESOURCE_SOURCE_KEY = "resource_source"
_RESOURCE_SOURCE_CALLER = "caller"
# Both markers are instructions to the binding, stripped from the response by _normalize_contexts.
_MARKER_KEYS = frozenset({_SUBJECT_SOURCE_KEY, _RESOURCE_SOURCE_KEY})


def _subject_source_parameters(evaluation: dict[str, Any]) -> dict[str, Any]:
    """The evaluation's context.parameters object, or `{}` when the context or parameters
    aren't objects - a malformed marker reads as no marker at all.
    """
    context = evaluation.get("context")
    parameters = context.get("parameters") if isinstance(context, dict) else None
    return parameters if isinstance(parameters, dict) else {}


def _subject_source(evaluation: dict[str, Any]) -> object:
    """Return the evaluation's context.parameters.subject_source marker, or None."""
    return _subject_source_parameters(evaluation).get(_SUBJECT_SOURCE_KEY)


def _complete_second_subject_types(data: dict[str, Any]) -> None:
    """Fill in a blank `subject.type` on a later evaluation whose `subject.id` is the gate
    evaluation's resource id, from that gate resource's type. The model (flash-lite, seen live)
    writes the second subject as `{"id": "alison"}` without the type; validation would reject
    that as blank and the retry rarely mends it. evaluations[0] already declares the same
    identity with its type (`{"type": "Customer", "id": "alison"}`), so the completion restates
    what the model itself produced - and _subject_from_prompt's marker-plus-gate check still
    decides whether that subject is trusted or overridden with the caller.
    """
    gate = _gate_resource(data)
    if gate is None:
        return
    gate_type, gate_id = gate
    for evaluation in data.get("evaluations", [])[1:]:
        if not isinstance(evaluation, dict):
            continue
        subject = evaluation.get("subject")
        if (
            _subject_source(evaluation) == _SUBJECT_SOURCE_PROMPT  # a second subject, never a search
            and isinstance(subject, dict)
            and _blank(subject.get("type"))
            and subject.get("id") == gate_id
        ):
            subject["type"] = gate_type


def _bind_caller_resources(data: dict[str, Any], *, subject_id: str | None, subject_type: str) -> None:
    """On an evaluation marked `resource_source: "caller"`, make the resource the caller: the
    caller's type, and the caller's id (none for an anonymous caller) - whatever the model put
    there, since the marker means the caller and nothing else. Runs before validation so a
    type the model left blank is filled in from the request rather than rejected.
    """
    for evaluation in data.get("evaluations", []):
        if not isinstance(evaluation, dict):
            continue
        if _subject_source_parameters(evaluation).get(_RESOURCE_SOURCE_KEY) != _RESOURCE_SOURCE_CALLER:
            continue
        resource = evaluation.get("resource")
        if resource is None:
            resource = {}
            evaluation["resource"] = resource
        elif not isinstance(resource, dict):
            continue  # malformed - left for validation to report, not repaired into the caller
        resource["type"] = subject_type
        if subject_id:
            resource["id"] = subject_id
        else:
            resource.pop("id", None)


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
