"""Schema fixtures shared by the integration tests."""

import json
from pathlib import Path
from typing import Any

SCHEMA_FIXTURE_PATH = Path(__file__).parent.parent / "test" / "fixtures" / "mock_ikg_schema.json"


def load_schema_fixture() -> dict[str, Any]:
    """The captured IKG schema example, as served by the platform."""
    return json.loads(SCHEMA_FIXTURE_PATH.read_text())


def policies(*grants: tuple[str, str, str]) -> list[dict[str, Any]]:
    """A GET /access/v1/policies listing granting each (subject type, action, resource type)."""
    return [
        {"policy": {"subject": {"type": s}, "actions": [a], "resource": {"type": r}}, "tags": []} for s, a, r in grants
    ]


# Grants the "buy AAPL" answer the integration tests mock, over schema_with_stocks().
STOCK_POLICIES = policies(("Customer", "buy", "Stock"))
# The rating every mocked answer with evaluations must carry.
RATING = '"explanation": "Customer buy Stock matches.", "confidence": "high"'


def schema_with_stocks() -> dict[str, Any]:
    """The shared schema fixture plus a Stock node and a Customer-buy->Stock edge, so the
    mocked "buy AAPL" answer passes the validation loop (TDD 4.8) instead of being retried.
    Kept out of the fixture file: the unit tests pin its exact vocabulary and rely on
    Stock/BUY being off-schema.
    """
    schema = load_schema_fixture()
    schema["graph"]["nodes"]["Stock"] = {"metadata": {"properties": {}, "node_count": 1}}
    schema["graph"]["edges"].append({"source": "Customer", "relation": "buy", "target": "Stock", "directed": True})
    return schema
