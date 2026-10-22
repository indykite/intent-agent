import json
from pathlib import Path
from typing import Any, get_args, get_origin

import pytest

from indykite.intent_agent.input_schema import (
    _Context,
    build_input_schema,
    digital_twin_types,
    policy_triples,
    render_schema_summary,
)
from indykite.intent_agent.subject_binding import _CONTEXT_KEY_TYPES

_SCHEMA_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_ikg_schema.json"


def _load_schema_fixture():
    """Load the real captured IKG schema example used for schema-shape tests."""
    return json.loads(_SCHEMA_FIXTURE_PATH.read_text())


def _policy(subject_type, actions, resource_type):
    """One GET /access/v1/policies entry granting `actions` to `subject_type` on `resource_type`."""
    return {
        "policy": {"subject": {"type": subject_type}, "actions": actions, "resource": {"type": resource_type}},
        "tags": [],
    }


# Policies over the mock schema's node types (Car, Company, Customer, _Application).
_POLICIES = [
    _policy("Customer", ["RENT", "OWN"], "Car"),
    _policy("Customer", ["WORKS_AT"], "Company"),
]


def _variant(field_schema: dict[str, Any], type_name: str) -> dict[str, Any]:
    """Return the `anyOf` branch (or the schema itself if not a union) whose type is `type_name`."""
    for variant in field_schema.get("anyOf", [field_schema]):
        if variant.get("type") == type_name:
            return variant
    msg = f"no {type_name!r} variant in {field_schema!r}"
    raise AssertionError(msg)


def _enum(field_schema):
    return _variant(field_schema, "string").get("enum")


_CUSTOMER_VOCABULARY = (["Customer"], ["OWN", "RENT", "WORKS_AT"], ["Car", "Company"])


@pytest.mark.parametrize(
    ("schema", "policies", "expected"),
    [
        pytest.param(_load_schema_fixture(), _POLICIES, _CUSTOMER_VOCABULARY, id="policy_vocabulary"),
        pytest.param(
            _load_schema_fixture(),
            [_policy("Customer", ["APPROVE_REFUND"], "Car")],
            (["Customer"], ["APPROVE_REFUND"], ["Car"]),
            id="edges_are_never_actions",
        ),
        pytest.param(
            _load_schema_fixture(),
            [*_POLICIES, _policy("Customer", ["FLY"], "Plane"), _policy("Pilot", ["FLY"], "Car")],
            _CUSTOMER_VOCABULARY,
            id="policy_over_a_type_the_schema_lacks_is_excluded",
        ),
        pytest.param(
            _load_schema_fixture(),
            [
                _policy("Customer", ["APPROVE_REFUND"], "Car"),
                _policy("Customer", ["APPROVE_REFUND", "ESCALATE"], "Company"),
            ],
            (["Customer"], ["APPROVE_REFUND", "ESCALATE"], ["Car", "Company"]),
            id="every_fetched_policys_actions",
        ),
        pytest.param(
            _load_schema_fixture(),
            [
                {"policy": {"subject": {"type": "Customer"}, "resource": {"type": "Car"}}},
                _policy("Customer", [123, "", " ", "ESCALATE"], "Car"),
                {"policy": {"actions": ["NO_TYPES"]}},
                {"policy": "?"},
                {"not_a_policy_key": True},
                "not-an-entry",
            ],
            (["Customer"], ["ESCALATE"], ["Car"]),
            id="malformed_policy_entries_ignored",
        ),
        pytest.param(_load_schema_fixture(), None, (None, None, None), id="no_policies_no_enums"),
        pytest.param({}, _POLICIES, (None, None, None), id="missing_graph_sections_no_enums"),
    ],
)
def test_build_input_schema_enums_come_from_policies(schema, policies, expected):
    """subject.type / action.name / resource.type enums, in that order (None = unconstrained)."""
    defs = build_input_schema(schema, policies)["$defs"]
    assert (
        _enum(defs["_Subject"]["properties"]["type"]),
        _enum(defs["_Action"]["properties"]["name"]),
        _enum(defs["_Resource"]["properties"]["type"]),
    ) == expected


def test_build_input_schema_allows_subject_id_for_the_second_subject_case():
    """subject.id is in the output schema so the LLM can set it for a second-subject evaluation."""
    schema = build_input_schema(_load_schema_fixture())
    assert "id" in schema["$defs"]["_Subject"]["properties"]


def test_build_input_schema_context_has_exactly_the_tdd_keys():
    """context exposes exactly parameters/filters/qualifiers, not an open object."""
    schema = build_input_schema(_load_schema_fixture())
    context_properties = schema["$defs"]["_Context"]["properties"]
    assert set(context_properties.keys()) == {"parameters", "filters", "qualifiers"}
    _variant(context_properties["filters"], "string")
    _variant(context_properties["parameters"], "object")
    _variant(context_properties["qualifiers"], "object")


def test_build_input_schema_top_level_is_attack_second_subject_explanation_evaluations_confidence():
    schema = build_input_schema(_load_schema_fixture())
    assert schema["required"] == ["evaluations"]
    assert list(schema["properties"]) == ["attack", "second_subject", "explanation", "evaluations", "confidence"]
    assert {"type": "string"} in schema["properties"]["explanation"]["anyOf"]
    assert {"type": "string", "enum": ["high", "medium", "low"]} in schema["properties"]["confidence"]["anyOf"]


def test_context_shape_check_covers_every_structured_output_context_key():
    """subject_binding's context check and the structured output's _Context list the same keys and types."""
    assert set(_CONTEXT_KEY_TYPES) == set(_Context.model_fields)
    for key, expected in _CONTEXT_KEY_TYPES.items():
        annotation = _Context.model_fields[key].annotation
        declared = {get_origin(variant) or variant for variant in get_args(annotation)}
        assert expected in declared, f"{key}: validator expects {expected}, structured output declares {annotation}"


def test_policy_triples_reads_each_granted_combination():
    assert policy_triples(_POLICIES, _load_schema_fixture()) == {
        ("Customer", "RENT", "Car"),
        ("Customer", "OWN", "Car"),
        ("Customer", "WORKS_AT", "Company"),
    }
    assert policy_triples(None, _load_schema_fixture()) == set()


def test_build_input_schema_drops_pydantic_only_keys():
    """`title`/`default` are stripped: Gemini's response_json_schema doesn't support `default`."""
    serialized = json.dumps(build_input_schema(_load_schema_fixture()))
    assert '"title"' not in serialized
    assert '"default"' not in serialized


def test_render_schema_summary_lists_node_types_edges_and_policies_sorted():
    schema = {
        "graph": {
            "nodes": {
                "Document": {
                    "metadata": {
                        "properties": {
                            "title": {"types": [{"type": "STRING NOT NULL"}]},
                            "external_id": {"types": [{"type": "STRING"}, {"type": "FLOAT"}, "?"]},
                            "pages": {"types": "?"},
                        },
                        "system_labels": [{"name": "Resource"}],
                    }
                },
                "User": {"metadata": {"properties": {"external_id": {}}, "system_labels": [{"name": "DigitalTwin"}]}},
                "Ghost": {},
            },
            "edges": [
                {"source": "User", "relation": "CAN_ACCESS", "target": "Document", "directed": True},
                {"source": "User", "relation": "CAN_ACCESS", "target": "Document"},  # duplicate relation entry
                {"source": "Document", "relation": "AUTHORED_BY", "target": "User"},
                {"source": "User", "relation": 7, "target": "Document"},  # malformed, skipped
                "not-an-edge",
            ],
        },
    }
    policies = [_policy("User", ["CAN_READ", "CAN_ACCESS"], "Document")]

    summary = render_schema_summary(schema, policies)

    assert summary.splitlines() == [
        "Node types, with their properties (only a type a policy below names may be a Subject or Resource):",
        "- Document: external_id:FLOAT|STRING, pages, title:STRING",
        "- Ghost: (no properties)",
        "- User (DigitalTwin): external_id",
        "",
        "Edges (how the data is linked - never action names), as source -RELATION-> target:",
        "- Document -AUTHORED_BY-> User",
        "- User -CAN_ACCESS-> Document",
        "",
        "Policies (the only valid combinations - every evaluation uses one), as subject -ACTION-> resource:",
        "- User -CAN_ACCESS-> Document",
        "- User -CAN_READ-> Document",
    ]


def test_render_schema_summary_tolerates_missing_or_malformed_graph_sections():
    assert "- (none)" in render_schema_summary({})
    summary = render_schema_summary({"graph": {"nodes": "?", "edges": "?"}})
    assert summary.count("- (none)") == len(["nodes", "edges", "policies"])


def test_graph_readers_tolerate_a_non_object_graph():
    """A non-object `graph` reads as an empty graph everywhere, never an AttributeError."""
    malformed = {"graph": "?"}
    assert render_schema_summary(malformed, _POLICIES).count("- (none)") == len(["nodes", "edges", "policies"])
    action_name = build_input_schema(malformed, _POLICIES)["$defs"]["_Action"]["properties"]["name"]
    assert "enum" not in _variant(action_name, "string")
    assert digital_twin_types(malformed) == []


def test_render_schema_summary_is_far_smaller_than_the_raw_document():
    """A real captured schema's summary fits well inside SCHEMA_PREVIEW_CHARS."""
    schema = json.loads((Path(__file__).parent / "fixtures" / "canbank_schema.json").read_text())
    raw = json.dumps(schema, indent=2)
    summary = render_schema_summary(schema)
    assert len(summary) < len(raw) / 5
    assert len(summary) < 12000  # noqa: PLR2004 - the Settings default for SCHEMA_PREVIEW_CHARS


def test_render_schema_summary_flags_platform_internal_types_instead_of_marking_them_subjects():
    """_Application carries the DigitalTwin label but is never what a prompt is about."""
    schema = {
        "graph": {
            "nodes": {"_Application": {"metadata": {"properties": {}, "system_labels": [{"name": "DigitalTwin"}]}}},
            "edges": [],
        },
    }
    assert "- _Application (platform-internal, never a Subject or Resource): (no properties)" in render_schema_summary(
        schema
    )
    assert "_Application (DigitalTwin)" not in render_schema_summary(schema)


def test_build_input_schema_requires_resource_id():
    """resource.id is required because the model fills required fields far more reliably; "" means many."""
    resource = build_input_schema(_load_schema_fixture())["$defs"]["_Resource"]
    assert resource["required"] == ["id"]
    assert resource["properties"]["id"]["type"] == "string"


def test_render_schema_summary_lists_only_policies_over_schema_node_types():
    schema = _load_schema_fixture()
    policies = [
        _policy("Customer", ["CHECK_MEMBERSHIP"], "Car"),
        _policy("Company", ["CAN_AUDIT"], "Car"),
        _policy("Company", ["CAN_AUDIT"], "Plane"),
        {"policy": "?"},
    ]

    with_policies = render_schema_summary(schema, policies)
    without = render_schema_summary(schema)

    heading = "Policies (the only valid combinations - every evaluation uses one), as subject -ACTION-> resource:"
    assert with_policies.endswith(f"{heading}\n- Company -CAN_AUDIT-> Car\n- Customer -CHECK_MEMBERSHIP-> Car")
    assert without.endswith(f"{heading}\n- (none)")
