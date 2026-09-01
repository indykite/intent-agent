"""Schema-derived subject-type validation and deterministic subject binding (TDD 4.7)."""

import json
import logging
import re
from typing import Any

from indykite.intent_agent.errors import SubjectTypeMismatchError, malformed_response_error, no_intent_error

logger = logging.getLogger(__name__)


def _node_types(schema: dict[str, Any]) -> set[str]:
    """Return every node type label defined in the given schema."""
    nodes = schema.get("graph", {}).get("nodes", {})
    if not isinstance(nodes, dict):
        return set()
    return set(nodes.keys())


def validate_subject_type(schema: dict[str, Any] | None, subject_type: str) -> None:
    """Raise if subject_type isn't one of the schema's node types."""
    if schema is None:
        return
    types = _node_types(schema)
    if subject_type not in types:
        message = f"SUBJECT_TYPE {subject_type!r} is not a node type in the schema."
        raise SubjectTypeMismatchError(message)


def _evaluation_unresolved(evaluation: object) -> bool:
    """True if the LLM attempted subject.type/action.name/resource.type but couldn't
    correlate it to the schema's enum vocabulary (input_schema.build_input_schema).

    Must run before _bind_subject, which unconditionally overwrites subject.type/id
    regardless of what the LLM produced. A key entirely absent (e.g. no `subject` for
    Access Evaluation, no `action` for Action Search - see SKILL.md) is a legitimate
    per-evaluation-kind omission, not a failure; a key present with an empty type/name
    inside it means the LLM tried and failed to correlate it.
    """
    if not isinstance(evaluation, dict):
        return True
    resource = evaluation.get("resource")
    if not isinstance(resource, dict) or not resource.get("type"):
        return True
    action = evaluation.get("action")
    if isinstance(action, dict) and not action.get("name"):
        return True
    subject = evaluation.get("subject")
    return isinstance(subject, dict) and not subject.get("type")


def normalize_intent_response(
    llm_text: str,
    original_prompt: str,
    *,
    subject_id: str | None,
    subject_type: str,
) -> str:
    """Ensure the response is valid JSON with prompt/evaluations, wrap it if needed.

    Assumes the schema is available - run_extraction checks that upfront and never
    reaches this far (or the LLM) otherwise. The LLM's output is constrained to
    {prompt, evaluations} an empty evaluations list is the only "couldn't extract"
    signal to handle here; Errors will not be LLM-supplied.
    """
    text = llm_text.strip()
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        text = json_match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.exception("LLM produced non-JSON output for prompt %r: %.500s", original_prompt, text)
        return json.dumps(malformed_response_error(original_prompt), indent=2, ensure_ascii=False)
    if not isinstance(data, dict):
        # A bare JSON array is a plausible LLM mistake (forgetting the {"prompt", "evaluations"}
        # wrapper) - use its items as the evaluations directly instead of nesting the whole
        # array as a single non-dict "evaluation" that _bind_subject would then skip.
        data = {"prompt": original_prompt, "evaluations": data if isinstance(data, list) else []}
    data.setdefault("prompt", original_prompt)
    evaluations = data.get("evaluations")
    if not evaluations or any(_evaluation_unresolved(evaluation) for evaluation in evaluations):
        return json.dumps(no_intent_error(original_prompt), indent=2, ensure_ascii=False)

    _bind_subject(data, subject_id=subject_id, subject_type=subject_type)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _bind_subject(data: dict[str, Any], *, subject_id: str | None, subject_type: str) -> None:
    """Force subject.id/subject.type on every evaluation from the validated token and
    deployment config, not whatever the LLM produced - closes the prompt-injection
    surface where the LLM could otherwise set the subject itself.

    TODO: Subject Search evaluations use subject.type as a search parameter, not the
    caller's own identity, and must not get this override - deferred to a later task.
    """
    for evaluation in data.get("evaluations", []):
        if not isinstance(evaluation, dict):
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
