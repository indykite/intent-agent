"""Structured error responses and startup-failure exceptions (TDD 4.7, 4.4)."""


class SubjectTypeMismatchError(Exception):
    """Raised when the caller-supplied subject_type isn't a node type in the schema.

    Caught per-request in run_extraction() and turned into subject_type_mismatch_error() -
    an invalid subject_type doesn't get fixed by retrying, but it must not take down an
    otherwise-healthy process either.
    """


class IntentValidationError(Exception):
    """Raised when the LLM's extraction output can't be accepted as-is: not JSON, a
    subject.type / action.name / resource.type that isn't in the cached schema (TDD 4.8),
    or a context whose filters / parameters / qualifiers have the wrong type.

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
    """Build the 4.7 error shape for when the schema can't be used to verify subject_type."""
    return error_response(
        prompt,
        "Data schema unavailable - cannot verify the supplied subject type against it.",
        "Retry once the data schema is available, or contact the administrator.",
    )


def schema_not_populated_error(prompt: str) -> dict[str, str]:
    """Build the 4.7 error shape for when the schema is fetched but the data model has no
    node types beyond the platform's own bootstrap node (_Application) - i.e. nothing has
    been modeled yet. Distinct from schema_unavailable_error (the fetch itself failed) and
    from subject_type_mismatch_error (real types exist, the supplied subject_type just isn't
    one of them) - there's nothing yet to validate subject_type against, so this is neither a
    fetch failure nor a mismatch.
    """
    return error_response(
        prompt,
        "Data schema is available but the data model hasn't been populated with any entity types yet.",
        "Retry once the data model has been populated, or contact the administrator.",
    )


def subject_type_mismatch_error(prompt: str) -> dict[str, str]:
    """Build the 4.7 error shape for when the caller-supplied subject_type isn't a valid type
    in the schema.

    Deliberately generic, unlike the log line for the same failure (which does include the
    schema's actual node type names): the caller of the intent-agent's API shouldn't learn
    the shape of the underlying data schema just by tripping this check.
    """
    return error_response(
        prompt,
        "subject_type is not a valid entity type for the current data schema.",
        "Contact the administrator to correct the subject_type supplied for this request.",
    )


def no_intent_error(prompt: str) -> dict[str, str]:
    """Build the 4.7 error shape for when no coherent intent could be extracted.

    Covers both the LLM's own empty evaluations list (its "nothing coherent" signal,
    SKILL.md) and the TDD 4.9 terminal failure: output still not JSON / off-schema after
    the retry budget. Deliberately generic, like subject_type_mismatch_error: the specific
    off-schema values reach the server-side log, but the caller shouldn't learn the
    schema's vocabulary from an error message. An empty evaluations list must never reach
    the caller as a bare success response.
    """
    return error_response(
        prompt,
        "The prompt doesn't match the Knowledge Graph Schema, or is unclear.",
        "Ask the prompter to clarify their request or ensure it's relevant to the system.",
    )


# The two ways subject_binding._vocabulary_reason reports a value it couldn't match to the data
# schema (off-schema, or left blank because nothing matched) - shared so validation_feedback_error
# can tell those apart from shape problems without guessing at the wording.
SCHEMA_MEMBERSHIP_SUFFIX = "is not defined in the data schema"
SCHEMA_BLANK_SUFFIX = "is missing or blank"
SCHEMA_VALUE_SUFFIXES = (SCHEMA_MEMBERSHIP_SUFFIX, SCHEMA_BLANK_SUFFIX)

_FEEDBACK_OBLIGATION = (
    'Return a corrected JSON object with the same {"evaluations"} shape, using only subject types, '
    "action names and resource types defined in the data schema, with context.filters a valid CEL "
    "expression over resource.<property> only, under 1000 characters, and context.parameters / "
    "context.qualifiers objects. "
    "If the prompt can't be mapped onto the schema, return an empty evaluations array instead."
)


def validation_feedback_error(prompt: str, errors: list[str]) -> dict[str, str]:
    """Build the 4.7 error shape fed back to the LLM as the follow-up message before a retry
    (TDD 4.9). `reason` lists what was wrong (every error, one per line - not just the first,
    or the LLM can only fix one thing per retry), not the schema's valid values: the LLM
    already holds the full schema from activate_skill, and the structured-output enum
    re-constrains the retry. The errors are split into two headed groups, values it couldn't
    match to the data schema (off-schema or left blank) and malformed shape (not JSON, a missing
    key, a mis-typed context), so the LLM isn't told to change a type when it's the shape that's
    wrong, or vice versa. `obligation` tells it what to return instead.
    """
    schema_errors = [error for error in errors if error.endswith(SCHEMA_VALUE_SUFFIXES)]
    shape_errors = [error for error in errors if not error.endswith(SCHEMA_VALUE_SUFFIXES)]
    groups = (
        ("Values not matched to the data schema:", schema_errors),
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
    """Build the 4.7 error shape for a timeout: a single LLM call past LLM_TIMEOUT_SECONDS, or
    the whole extraction (every call, tool round and retry) past REQUEST_TIMEOUT_SECONDS. The
    server log tells the two apart; the caller's remedy is the same either way.
    """
    return error_response(
        prompt,
        "The model did not respond in time.",
        "Retry the request, or contact the administrator if the issue persists.",
    )
