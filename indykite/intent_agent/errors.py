"""Structured error responses and startup-failure exceptions."""


class SubjectTypeMismatchError(Exception):
    """Raised when the caller-supplied subject_type isn't a node type in the schema.

    Fails only that request (as subject_type_mismatch_error), never the process.
    """


class IntentValidationError(Exception):
    """Raised when the LLM's extraction output is not JSON, off-schema or mis-shaped.

    run_extraction feeds `errors` back to the LLM and retries up to VALIDATION_MAX_RETRIES times.
    """

    def __init__(self, errors: list[str]) -> None:
        """Keep the individual messages so they can be fed back to the LLM one per line."""
        super().__init__("; ".join(errors))
        self.errors = errors


class InvalidAppAgentKeyError(Exception):
    """Raised when IK_APP_AGENT_KEY isn't a well-formed JWT with a `sub` claim; fails startup."""


def error_response(prompt: str, reason: str, obligation: str) -> dict[str, str]:
    """Build the structured error shape {prompt, error, reason, obligation} used instead of evaluations."""
    return {
        "prompt": prompt,
        "error": "Unable to extract intent from prompt.",
        "reason": reason,
        "obligation": obligation,
    }


def schema_unavailable_error(prompt: str) -> dict[str, str]:
    """Build the error for when the schema can't be used to verify subject_type."""
    return error_response(
        prompt,
        "Data schema unavailable - cannot verify the supplied subject type against it.",
        "Retry once the data schema is available, or contact the administrator.",
    )


def policy_unavailable_error(prompt: str) -> dict[str, str]:
    """Build the error for when no access policy exists for the caller's subject type."""
    return error_response(
        prompt,
        "Access policies for the supplied subject type are unavailable.",
        "Retry once access policies are available, or contact the administrator.",
    )


def schema_not_populated_error(prompt: str) -> dict[str, str]:
    """Build the error for a fetched schema with no node types beyond the bootstrap _Application."""
    return error_response(
        prompt,
        "Data schema is available but the data model hasn't been populated with any entity types yet.",
        "Retry once the data model has been populated, or contact the administrator.",
    )


def subject_type_mismatch_error(prompt: str) -> dict[str, str]:
    """Build the error for a caller-supplied subject_type that isn't a type in the schema.

    Deliberately generic so the caller can't learn the schema's node types from it.
    """
    return error_response(
        prompt,
        "subject_type is not a valid entity type for the current data schema.",
        "Contact the administrator to correct the subject_type supplied for this request.",
    )


def no_intent_error(prompt: str) -> dict[str, str]:
    """Build the error for no coherent intent: an empty evaluations list or retries exhausted.

    Deliberately generic so the caller can't learn the schema's vocabulary from it.
    """
    return error_response(
        prompt,
        "The prompt doesn't match the Knowledge Graph Schema, or is unclear.",
        "Ask the prompter to clarify their request or ensure it's relevant to the system.",
    )


# Error suffixes shared with subject_binding so validation_feedback_error can group errors by kind.
SCHEMA_MEMBERSHIP_SUFFIX = "is not defined in the data schema"
SCHEMA_BLANK_SUFFIX = "is missing or blank"
POLICY_MEMBERSHIP_SUFFIX = "is not named by any access policy"
SCHEMA_RELATIONSHIP_SUFFIX = "is not a combination any access policy grants"
SCHEMA_VALUE_SUFFIXES = (
    SCHEMA_MEMBERSHIP_SUFFIX,
    SCHEMA_BLANK_SUFFIX,
    POLICY_MEMBERSHIP_SUFFIX,
    SCHEMA_RELATIONSHIP_SUFFIX,
)

_FEEDBACK_OBLIGATION = (
    "Return a corrected JSON object with the same keys: second_subject (or null), explanation, "
    "evaluations and confidence - explanation and confidence are required whenever evaluations is "
    "not empty. Every evaluation's subject type, action name and resource type together form one "
    "subject -ACTION-> resource combination listed under Policies; an Action Search keeps its "
    "action omitted. context.filters is a valid CEL expression over resource.<property> only, under "
    "1000 characters, and context.parameters / context.qualifiers are objects. A graph edge is never "
    "an action name. If no policy covers the prompt, return an empty evaluations array with an "
    "explanation instead."
)


def validation_feedback_error(prompt: str, errors: list[str]) -> dict[str, str]:
    """Build the error fed back to the LLM before a retry, listing every error.

    Errors are grouped into schema/policy value mismatches and malformed shape, so the LLM
    knows which kind of fix each needs.
    """
    schema_errors = [error for error in errors if error.endswith(SCHEMA_VALUE_SUFFIXES)]
    shape_errors = [error for error in errors if not error.endswith(SCHEMA_VALUE_SUFFIXES)]
    groups = (
        ("Values not matched to the data schema or access policies:", schema_errors),
        ("Malformed shape:", shape_errors),
    )
    sections = [
        f"{heading}\n" + "\n".join(f"- {error}" for error in group_errors)
        for heading, group_errors in groups
        if group_errors
    ]
    return error_response(
        prompt, "The previous response can't be accepted.\n" + "\n".join(sections), _FEEDBACK_OBLIGATION
    )


def llm_timeout_error(prompt: str) -> dict[str, str]:
    """Build the error for a timeout during the LLM phase (per-call or whole-request deadline).

    A deadline hit during the schema or policy fetch is request_timeout_error instead.
    """
    return error_response(
        prompt,
        "The model did not respond in time.",
        "Retry the request, or contact the administrator if the issue persists.",
    )


def request_timeout_error(prompt: str) -> dict[str, str]:
    """Build the error for REQUEST_TIMEOUT_SECONDS expiring during the schema or policy fetch."""
    return error_response(
        prompt,
        "The request could not be completed within the time budget.",
        "Retry the request, or contact the administrator if the issue persists.",
    )
