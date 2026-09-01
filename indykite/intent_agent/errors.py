"""Structured error responses and startup-failure exceptions (TDD 4.7, 4.4)."""


class SubjectTypeMismatchError(Exception):
    """Raised when the configured SUBJECT_TYPE isn't a node type in the schema.

    Caught per-request in run_extraction() and turned into subject_type_mismatch_error() -
    a misconfigured SUBJECT_TYPE doesn't get fixed by retrying, but it must not take down an
    otherwise-healthy process either.
    """


class InvalidAppAgentKeyError(Exception):
    """Raised when IK_APP_AGENT_KEY isn't a well-formed JWT with a `sub` claim.

    A malformed app-agent key is a real configuration error and must fail the process
    loudly at startup, not silently degrade (e.g. to a hashed cache key) while continuing
    to attempt a schema fetch with a broken credential.
    """


def error_response(prompt: str, reason: str, obligation: str) -> dict[str, str]:
    """Build the TDD's structured error shape (4.7): {"error", "reason", "obligation"}
    instead of "evaluations".
    """
    return {
        "prompt": prompt,
        "error": "Unable to extract intent from prompt.",
        "reason": reason,
        "obligation": obligation,
    }


def schema_unavailable_error(prompt: str) -> dict[str, str]:
    """Build the 4.7 error shape for when the schema can't be used to verify subject.type."""
    return error_response(
        prompt,
        "Data schema unavailable - cannot verify the configured subject type against it.",
        "Retry once the data schema is available, or contact the administrator.",
    )


def schema_not_populated_error(prompt: str) -> dict[str, str]:
    """Build the 4.7 error shape for when the schema is fetched but the data model has no
    node types beyond the platform's own bootstrap node (_Application) - i.e. nothing has
    been modeled yet. Distinct from schema_unavailable_error (the fetch itself failed) and
    from subject_type_mismatch_error (real types exist, SUBJECT_TYPE just isn't one of them) -
    there's nothing yet to validate SUBJECT_TYPE against, so this is neither a fetch failure
    nor a misconfiguration.
    """
    return error_response(
        prompt,
        "Data schema is available but the data model hasn't been populated with any entity types yet.",
        "Retry once the data model has been populated, or contact the administrator.",
    )


def subject_type_mismatch_error(prompt: str) -> dict[str, str]:
    """Build the 4.7 error shape for when SUBJECT_TYPE isn't a valid type in the schema.

    Deliberately generic, unlike the log line for the same failure (which does include the
    schema's actual node type names): the caller of the intent-agent's API shouldn't learn
    the shape of the underlying data schema just by tripping a config error.
    """
    return error_response(
        prompt,
        "SUBJECT_TYPE is not a valid entity type for the current data schema.",
        "Contact the administrator to correct the SUBJECT_TYPE configuration.",
    )


def no_intent_error(prompt: str) -> dict[str, str]:
    """Build the 4.7 error shape for when no coherent intent could be extracted.

    Fallback for when the LLM returns an empty evaluations list without its own
    reason/obligation (SKILL.md instructs it to supply those directly instead) -
    an empty evaluations list must never reach the caller as a bare success response.
    """
    return error_response(
        prompt,
        "The prompt doesn't match the Knowledge Graph Schema, or is unclear.",
        "Ask the prompter to clarify their request or ensure it's relevant to the system.",
    )


def malformed_response_error(prompt: str) -> dict[str, str]:
    """Build the 4.7 error shape for when the LLM's output can't be parsed as JSON at all."""
    return error_response(
        prompt,
        "The model's response could not be parsed as JSON.",
        "Retry the request, or contact the administrator if the issue persists.",
    )
