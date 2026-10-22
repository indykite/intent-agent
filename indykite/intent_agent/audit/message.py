"""The audit record: one per request, with `prompt` and `evaluations` recorded verbatim.

`message` always carries the same text the caller received.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

# Same limit as the Agent Gateway: a longer X-Request-ID is ignored, not truncated.
_MAX_REQUEST_ID_LENGTH = 40
_REQUEST_ID_HEADER = "x-request-id"


class Outcome(StrEnum):
    """The audit record's `outcome` vocabulary."""

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
    policy_version: str
    llm_provider: str
    llm_model: str
    latency_ms: int
    validation_retries: int
    # None = no policy fetch attempted (request failed earlier); False = fetched successfully.
    policy_fetch_failed: bool | None
    outcome: Outcome
    message: str

    def to_json(self) -> str:
        """Serialize as one line of JSON (no trailing newline)."""
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)

    def to_row(self) -> list[str]:
        """Render every field as a string in field order, nested values as compact JSON."""
        return [_cell(value) for value in self.model_dump(mode="json").values()]


FIELD_NAMES: tuple[str, ...] = tuple(AuditMessage.model_fields)


def _cell(value: object) -> str:
    """Stringify one field value for CSV/TXT output; None renders as ""."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list | dict):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def utc_timestamp() -> str:
    """Return now as an RFC 3339 UTC timestamp with sub-second precision and a `Z` suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def schema_version(schema: Mapping[str, Any] | None) -> str:
    """Return a key-order-independent SHA-256 of the JGF schema, or "" when there is none."""
    if schema is None:
        return ""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def policy_version(policies: Sequence[Mapping[str, Any]] | None) -> str:
    """Return a SHA-256 of the policies independent of key and entry order, or "" when there are none."""
    if not policies:
        return ""
    canonical_entries = sorted(
        json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for policy in policies
    )
    canonical = json.dumps(canonical_entries, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def request_id_from_headers(headers: Mapping[str, str]) -> str:
    """Return the caller's X-Request-ID if non-blank and at most 40 characters, else a new UUID."""
    for name, value in headers.items():
        if name.lower() == _REQUEST_ID_HEADER:
            candidate = value.strip()
            if candidate and len(candidate) <= _MAX_REQUEST_ID_LENGTH:
                return candidate
            break
    return str(uuid4())
