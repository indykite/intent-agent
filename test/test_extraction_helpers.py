"""Helpers and constants shared by test_agent.py and test_subject_binding.py."""

import json
from pathlib import Path

# Fixture schema vocabulary: nodes Car/Company/Customer/_Application, edges OWN/RENT/WORKS_AT.
_ON_SCHEMA_ANSWER = (
    '{"prompt": "rent kitt", "evaluations": [{"action": {"name": "RENT"}, "resource": {"type": "Car", "id": "kitt"}}]}'
)

_OFF_SCHEMA_ANSWER = (
    '{"prompt": "buy AAPL", "evaluations": [{"action": {"name": "BUY"}, "resource": {"type": "Stock", "id": "AAPL"}}]}'
)

_SCHEMA_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_ikg_schema.json"


def _policy(subject_type, action, resource_type):
    """One `GET /access/v1/policies` entry granting subject_type `action` on resource_type."""
    return {
        "policy": {"subject": {"type": subject_type}, "actions": [action], "resource": {"type": resource_type}},
        "tags": [],
    }


# The test caller's (Customer) policies over the fixture schema, one per Customer edge.
_MOCK_POLICIES = [
    _policy("Customer", "RENT", "Car"),
    _policy("Customer", "OWN", "Car"),
    _policy("Customer", "WORKS_AT", "Company"),
]

_EMPTY_SCHEMA_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_ikg_schema_empty.json"


def _load_schema_fixture():
    """Load the real captured IKG schema example used for schema-shape tests."""
    return json.loads(_SCHEMA_FIXTURE_PATH.read_text())


def _load_empty_schema_fixture():
    """Load a captured empty-environment schema response (only the _Application node)."""
    return json.loads(_EMPTY_SCHEMA_FIXTURE_PATH.read_text())


_RATING = {"explanation": "RENT on Car is the caller's policy for renting.", "confidence": "high"}


def _rated(llm_text):
    """`llm_text` with _RATING added to an unrated {evaluations} answer; anything else unchanged."""
    try:
        data = json.loads(llm_text)
    except (TypeError, json.JSONDecodeError):
        return llm_text
    if not isinstance(data, dict) or "evaluations" not in data or not _RATING.keys().isdisjoint(data):
        return llm_text
    return json.dumps({**data, **_RATING})


def _stamped(context):
    """`context` with the `parameters` object the system gives every evaluation."""
    return {**context, "parameters": context.get("parameters", {})}
