"""Structured error responses and startup-failure exceptions (TDD 4.7, 4.4)."""


class IntentValidationError(Exception):
    """Raised when the LLM's extraction output can't be accepted as-is: not JSON, or a
    subject.type / action.name / resource.type that isn't in the cached schema (TDD 4.8).

    Caught per-attempt in IntentAgent.run_extraction(), which feeds `errors` back to the LLM
    as validation_feedback_error() and retries (TDD 4.9) - up to VALIDATION_MAX_RETRIES times -
    before giving up with no_intent_error().
    """

    def __init__(self, errors: list[str]) -> None:
        """Keep the individual messages so they can be fed back to the LLM one per line."""
        super().__init__("; ".join(errors))
        self.errors = errors


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
    """Build the 4.7 error shape for when the schema can't be used to map entity types."""
    return error_response(
        prompt,
        "Data schema unavailable - cannot map the prompt to entity types.",
        "Retry once the data schema is available, or contact the administrator.",
    )


def schema_not_populated_error(prompt: str) -> dict[str, str]:
    """Build the 4.7 error shape for when the schema is fetched but the data model has no
    node types beyond the platform's own bootstrap node (_Application) - i.e. nothing has
    been modeled yet. Distinct from schema_unavailable_error (the fetch itself failed) -
    there's nothing yet to map entity types against, but the schema itself isn't unreachable.
    """
    return error_response(
        prompt,
        "Data schema is available but the data model hasn't been populated with any entity types yet.",
        "Retry once the data model has been populated, or contact the administrator.",
    )


def no_intent_error(prompt: str) -> dict[str, str]:
    """Build the 4.7 error shape for when no coherent intent could be extracted.

    Covers both the LLM's own empty evaluations list (its "nothing coherent" signal,
    SKILL.md) and the TDD 4.9 terminal failure: output still not JSON / off-schema after
    the retry budget. Deliberately generic: the specific off-schema values reach the
    server-side log, but the caller shouldn't learn the schema's vocabulary from an error
    message. An empty evaluations list must never reach the caller as a bare success
    response.
    """
    return error_response(
        prompt,
        "The prompt doesn't match the Knowledge Graph Schema, or is unclear.",
        "Ask the prompter to clarify their request or ensure it's relevant to the system.",
    )


_FEEDBACK_OBLIGATION = (
    'Return a corrected JSON object with the same {"prompt", "evaluations"} shape, using only '
    "subject types, action names and resource types defined in the data schema. If the prompt "
    "can't be mapped onto the schema, return an empty evaluations array instead."
)


def validation_feedback_error(prompt: str, errors: list[str]) -> dict[str, str]:
    """Build the 4.7 error shape fed back to the LLM as the follow-up message before a retry
    (TDD 4.9). `reason` lists what was wrong (every error, one per line - not just the first,
    or the LLM can only fix one thing per retry), not the schema's valid values: the LLM
    already holds the full schema from activate_skill, and the structured-output enum
    re-constrains the retry. `obligation` tells it what to return instead.
    """
    return error_response(
        prompt,
        "The previous response doesn't conform to the data schema:\n" + "\n".join(f"- {error}" for error in errors),
        _FEEDBACK_OBLIGATION,
    )


def llm_timeout_error(prompt: str) -> dict[str, str]:
    """Build the 4.7 error shape for a timeout: a single LLM call past LLM_TIMEOUT_SECONDS, or
    the whole extraction (every call, tool round and retry) past REQUEST_TIMEOUT_SECONDS. The
    server log tells the two apart; the caller's remedy is the same either way.
    """
    return error_response(
        prompt,
        "The model did not respond in time.",
        "Retry the request, or contact the administrator if the issue persists.",
    )
