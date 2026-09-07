"""Schema fixtures shared by the integration tests."""

import json
from pathlib import Path
from typing import Any

SCHEMA_FIXTURE_PATH = Path(__file__).parent.parent / "test" / "fixtures" / "mock_ikg_schema.json"


def load_schema_fixture() -> dict[str, Any]:
    """The captured IKG schema example, as served by the platform."""
    return json.loads(SCHEMA_FIXTURE_PATH.read_text())


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
