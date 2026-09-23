import json
import re
from datetime import datetime

from indykite.intent_agent.audit.message import (
    FIELD_NAMES,
    Outcome,
    policy_version,
    request_id_from_headers,
    schema_version,
    utc_timestamp,
)

# TDD 4.9's field table, in its order, plus ENG-9671's policy_version/policy_fetch_failed - a
# deliberate, TDD-owner-confirmed pair of additions alongside their schema_version/(no failure
# counterpart) siblings, not a departure from "exactly these, nothing else" otherwise.
_TDD_FIELDS = (
    "timestamp",
    "request_id",
    "subject_id",
    "subject_type",
    "requesting_agent_id",
    "context_id",
    "prompt",
    "evaluations",
    "schema_version",
    "policy_version",
    "llm_provider",
    "llm_model",
    "latency_ms",
    "validation_retries",
    "policy_fetch_failed",
    "outcome",
    "message",
)
_MAX_REQUEST_ID_LENGTH = 40
_UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def test_record_has_exactly_the_tdd_fields_in_order(audit_message):
    """The wire shape is the TDD 4.9 table verbatim: same names, same order, no IAG extras."""
    assert FIELD_NAMES == _TDD_FIELDS
    assert tuple(json.loads(audit_message().to_json())) == _TDD_FIELDS


def test_outcome_vocabulary_is_the_tdd_one(audit_message):
    """`outcome` is one of the four TDD values, serialized lowercase."""
    assert {outcome.value for outcome in Outcome} == {"success", "validation_failed", "unmatched", "error"}
    assert json.loads(audit_message(outcome=Outcome.VALIDATION_FAILED).to_json())["outcome"] == "validation_failed"


def test_to_json_keeps_evaluations_structured(audit_message):
    """In JSON output evaluations stay a nested array, not a stringified blob."""
    payload = json.loads(audit_message().to_json())
    assert payload["evaluations"] == [{"action": {"name": "RENT"}, "resource": {"type": "Car", "id": "kitt"}}]
    assert payload["latency_ms"] == 42  # noqa: PLR2004 - the fixture's value


def test_to_row_flattens_every_field_to_a_string(audit_message):
    """CSV/TXT need one string per field - evaluations become compact JSON, numbers plain."""
    row = audit_message().to_row()
    assert len(row) == len(_TDD_FIELDS)
    assert all(isinstance(cell, str) for cell in row)
    assert row[_TDD_FIELDS.index("evaluations")] == '[{"action":{"name":"RENT"},"resource":{"type":"Car","id":"kitt"}}]'
    assert row[_TDD_FIELDS.index("latency_ms")] == "42"
    assert row[_TDD_FIELDS.index("outcome")] == "success"


def test_to_row_renders_unattempted_policy_fetch_as_blank_not_the_word_none(audit_message):
    """policy_fetch_failed=None (no policy fetch attempted this request, review) must render as
    "" in CSV/TXT output - the same "not applicable" sentinel schema_version/policy_version
    already use - not str(None) ("None" as literal text), which would be the only non-"True"/
    "False" value in that column and could break a downstream consumer expecting a strict bool.
    """
    row = audit_message(policy_fetch_failed=None).to_row()
    assert row[_TDD_FIELDS.index("policy_fetch_failed")] == ""


def test_utc_timestamp_is_rfc3339_utc_with_fraction():
    """RFC 3339 / ISO 8601, UTC (`Z`), sub-second precision - the IAG's RFC3339Nano shape."""
    stamp = utc_timestamp()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", stamp)
    assert datetime.fromisoformat(stamp).utcoffset().total_seconds() == 0


def test_schema_version_is_canonical_and_stable():
    """Key order must not change the hash; content must; no schema hashes to empty."""
    schema_a = {"graph": {"nodes": {"Car": {}, "Customer": {}}, "edges": []}}
    schema_b = {"graph": {"edges": [], "nodes": {"Customer": {}, "Car": {}}}}
    schema_c = {"graph": {"nodes": {"Car": {}}, "edges": []}}
    assert schema_version(schema_a) == schema_version(schema_b)
    assert schema_version(schema_a) != schema_version(schema_c)
    assert re.fullmatch(r"[0-9a-f]{64}", schema_version(schema_a))
    assert schema_version(None) == ""


def test_policy_version_is_canonical_and_stable_regardless_of_entry_order():
    """Key order within an entry, and the entries' own order in the list, must not change the
    hash (review: sort_keys=True alone only canonicalizes dict keys, not list order - the same
    policy state fetched in a different order must still hash identically); content must;
    empty/None both hash to "".
    """
    policy_a = {"policy": {"actions": ["RENT"], "resource": {"type": "Car"}}, "tags": []}
    policy_b = {"policy": {"actions": ["OWN"], "resource": {"type": "Car"}}, "tags": []}
    policies_ab = [policy_a, policy_b]
    policies_ba = [policy_b, policy_a]
    policy_a_reordered_keys = {"tags": [], "policy": {"resource": {"type": "Car"}, "actions": ["RENT"]}}
    policies_different_content = [policy_a]

    assert policy_version(policies_ab) == policy_version(policies_ba)
    assert policy_version([policy_a_reordered_keys, policy_b]) == policy_version(policies_ab)
    assert policy_version(policies_ab) != policy_version(policies_different_content)
    assert re.fullmatch(r"[0-9a-f]{64}", policy_version(policies_ab))
    assert policy_version(None) == ""
    assert policy_version([]) == ""


def test_request_id_prefers_a_sane_incoming_header():
    """A well-formed X-Request-ID (any header case) is kept so the record correlates with the
    caller's own logs; blank or over-long values fall back to a generated UUID, as in the IAG.
    """
    assert request_id_from_headers({"X-Request-ID": "abc-123"}) == "abc-123"
    assert request_id_from_headers({"x-request-id": " abc-123 "}) == "abc-123"
    assert request_id_from_headers({"x-request-id": "a" * _MAX_REQUEST_ID_LENGTH}) == "a" * _MAX_REQUEST_ID_LENGTH
    assert _UUID_PATTERN.match(request_id_from_headers({"x-request-id": "a" * (_MAX_REQUEST_ID_LENGTH + 1)}))
    assert _UUID_PATTERN.match(request_id_from_headers({"x-request-id": "   "}))
    assert _UUID_PATTERN.match(request_id_from_headers({}))


def test_request_ids_are_unique_when_generated():
    """Two requests without a header must not share a record id."""
    assert request_id_from_headers({}) != request_id_from_headers({})
