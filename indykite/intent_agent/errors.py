"""Structured error responses and startup-failure exceptions (TDD 4.7, 4.4)."""


class SubjectTypeMismatchError(Exception):
    """Raised when the configured SUBJECT_TYPE isn't a DigitalTwin-labelled node type in the schema."""


class SchemaUnavailableError(Exception):
    """Raised when the data schema can't be fetched at cold start.

    Per the TDD (4.4): the process must exit rather than serve responses with
    fallback or hardcoded types.
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
