"""Schema-derived subject-type validation and deterministic subject binding (TDD 4.7)."""

import json
import logging
import re
from typing import Any

from indykite.intent_agent.errors import SubjectTypeMismatchError, malformed_response_error, no_intent_error

logger = logging.getLogger(__name__)


def _digital_twin_types(schema: dict[str, Any]) -> set[str]:
    """Return the node type names labelled DigitalTwin in the given schema."""
    nodes = schema.get("graph", {}).get("nodes", {})
    if not isinstance(nodes, dict):
        return set()
    types = set()
    for type_name, node_def in nodes.items():
        system_labels = node_def.get("metadata", {}).get("system_labels", [])
        if any(isinstance(label, dict) and label.get("name") == "DigitalTwin" for label in system_labels):
            types.add(type_name)
    return types


def validate_subject_type(schema: dict[str, Any] | None, subject_type: str) -> None:
    """Raise if subject_type isn't one of the schema's DigitalTwin-labelled node types."""
    if schema is None:
        return
    types = _digital_twin_types(schema)
    if subject_type not in types:
        message = (
            f"SUBJECT_TYPE {subject_type!r} is not a DigitalTwin-labelled node type "
            f"in the schema (found: {sorted(types)})"
        )
        raise SubjectTypeMismatchError(message)


def normalize_intent_response(
    llm_text: str,
    original_prompt: str,
    *,
    subject_id: str | None,
    subject_type: str,
) -> str:
    """Ensure the response is valid JSON with prompt/evaluations, wrap it if needed.

    Assumes the schema is available - run_extraction checks that upfront and never
    reaches this far (or the LLM) otherwise.
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
    if "error" in data:
        return json.dumps(data, indent=2, ensure_ascii=False)
    if not data.get("evaluations"):
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
