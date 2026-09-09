"""The audit record (TDD 4.9): one per request, with exactly the TDD's fields.

`AuditMessage` follows the TDD rather than the Agent Gateway's `audit.Message`: the field names
are the TDD's (snake_case, `outcome`/`message` rather than `decision`/`reason`) and `prompt` /
`evaluations` are recorded verbatim. What *is* borrowed from the IAG is the `message` rule: it
carries the same text the caller received, so a record and the response it audits can't disagree.
"""

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

# The IAG applies this to the X-Request-ID it propagates into its logs (pkg/logging/fields):
# anything longer is treated as not a request id at all rather than truncated.
_MAX_REQUEST_ID_LENGTH = 40
_REQUEST_ID_HEADER = "x-request-id"


class Outcome(StrEnum):
    """The TDD 4.9 `outcome` vocabulary."""

    SUCCESS = "success"
    VALIDATION_FAILED = "validation_failed"
    UNMATCHED = "unmatched"
    ERROR = "error"


class AuditMessage(BaseModel):
    """One audit record. Field order here is the column order for CSV and TXT output."""

    model_config = ConfigDict(frozen=True)

    timestamp: str
    request_id: str
    subject_id: str
    subject_type: str
    requesting_agent_id: str
    context_id: str
    prompt: str
    evaluations: list[dict[str, Any]]
    schema_version: str
    llm_provider: str
    llm_model: str
    latency_ms: int
    validation_retries: int
    outcome: Outcome
    message: str

    def to_json(self) -> str:
        """Serialize as one line of JSON (no trailing newline)."""
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)

    def to_row(self) -> list[str]:
        """Render every field as a string, in field order - `evaluations` as compact JSON, since
        a nested structure has to fit in one CSV cell / one `key=value` pair.
        """
        return [_cell(value) for value in self.model_dump(mode="json").values()]


FIELD_NAMES: tuple[str, ...] = tuple(AuditMessage.model_fields)


def _cell(value: object) -> str:
    """Stringify one field value for CSV/TXT output."""
    if isinstance(value, str):
        return value
    if isinstance(value, list | dict):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def utc_timestamp() -> str:
    """Now, as an RFC 3339 UTC timestamp with sub-second precision and a `Z` suffix - the same
    shape as the IAG's `time.RFC3339Nano` (which is valid ISO 8601, as the TDD asks for).
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def schema_version(schema: Mapping[str, Any] | None) -> str:
    """Hash of the cached JGF schema (TDD: `schema_version` = "hash of cached JGF"), canonical so
    the same schema always hashes the same regardless of key order. Empty when no schema was
    available for the request.
    """
    if schema is None:
        return ""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def request_id_from_headers(headers: Mapping[str, str]) -> str:
    """Use the caller's X-Request-ID when it looks like one (non-blank, at most 40 characters -
    the IAG's own check), otherwise generate a UUID - the TDD's server-generated request_id.
    """
    for name, value in headers.items():
        if name.lower() == _REQUEST_ID_HEADER:
            candidate = value.strip()
            if candidate and len(candidate) <= _MAX_REQUEST_ID_LENGTH:
                return candidate
            break
    return str(uuid4())
