import json
from pathlib import Path
from typing import Any, get_args, get_origin

import pytest

from indykite.intent_agent.input_schema import _Context, build_input_schema
from indykite.intent_agent.subject_binding import _CONTEXT_KEY_TYPES

_SCHEMA_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_ikg_schema.json"


def _load_schema_fixture():
    """Load the real captured IKG schema example used for schema-shape tests."""
    return json.loads(_SCHEMA_FIXTURE_PATH.read_text())


def _variant(field_schema: dict[str, Any], type_name: str) -> dict[str, Any]:
    """Return the `anyOf` branch of `field_schema` matching `type_name` (or itself, if
    `field_schema` isn't a nullable/union field to begin with). Raises if none matches.
    """
    for variant in field_schema.get("anyOf", [field_schema]):
        if variant.get("type") == type_name:
            return variant
    msg = f"no {type_name!r} variant in {field_schema!r}"
    raise AssertionError(msg)


def test_build_input_schema_enumerates_all_node_types_as_resource_types():
    """resource.type must enumerate every node label (TDD 4.6), not just DigitalTwin ones."""
    schema = build_input_schema(_load_schema_fixture())
    resource_type = schema["$defs"]["_Resource"]["properties"]["type"]
    assert _variant(resource_type, "string")["enum"] == ["Car", "Company", "Customer", "_Application"]


def test_build_input_schema_enumerates_edge_relations_as_action_names():
    """action.name must enumerate every distinct edge relation in the schema (TDD 4.6)."""
    schema = build_input_schema(_load_schema_fixture())
    action_name = schema["$defs"]["_Action"]["properties"]["name"]
    assert _variant(action_name, "string")["enum"] == ["OWN", "RENT", "WORKS_AT"]


def test_build_input_schema_enumerates_all_node_types_as_subject_types():
    """subject.type must enumerate every node label, same as resource.type - SUBJECT_TYPE is
    no longer restricted to DigitalTwin-labelled types (ENG-9647), and the LLM's subject.type
    enum follows the same rule.
    """
    schema = build_input_schema(_load_schema_fixture())
    subject_type = schema["$defs"]["_Subject"]["properties"]["type"]
    assert _variant(subject_type, "string")["enum"] == ["Car", "Company", "Customer", "_Application"]


def test_build_input_schema_allows_subject_id_for_the_second_subject_case():
    """subject.id must be part of the LLM's output schema - the second-subject case
    (SKILL.md) requires the LLM to set it explicitly for a marked evaluation.
    subject_binding._bind_subject still overrides it on every other evaluation regardless
    of what the LLM puts here (TDD 4.7).
    """
    schema = build_input_schema(_load_schema_fixture())
    assert "id" in schema["$defs"]["_Subject"]["properties"]


def test_build_input_schema_context_has_exactly_the_tdd_keys():
    """context must expose exactly parameters/filters/qualifiers (TDD 4.6), not an open object."""
    schema = build_input_schema(_load_schema_fixture())
    context_properties = schema["$defs"]["_Context"]["properties"]
    assert set(context_properties.keys()) == {"parameters", "filters", "qualifiers"}
    _variant(context_properties["filters"], "string")
    _variant(context_properties["parameters"], "object")
    _variant(context_properties["qualifiers"], "object")


def test_build_input_schema_top_level_requires_only_evaluations():
    """The LLM's output shape is deliberately just {evaluations} - no error shape (our own code
    builds the error response, subject_binding.py) and no prompt echo (normalize_intent_response
    sets it from the incoming prompt; asking the model for it only costs output tokens).
    """
    schema = build_input_schema(_load_schema_fixture())
    assert schema["required"] == ["evaluations"]
    assert set(schema["properties"].keys()) == {"evaluations"}


def test_context_shape_check_covers_every_structured_output_context_key():
    """subject_binding's post-assembly context check is the belt-and-braces behind the
    structured output's _Context (the enum/types are a hint to Gemini, not a guarantee), so
    the two must list the same keys with the same expected types - a key added to one and not
    the other would silently go unchecked.
    """
    assert set(_CONTEXT_KEY_TYPES) == set(_Context.model_fields)
    for key, expected in _CONTEXT_KEY_TYPES.items():
        annotation = _Context.model_fields[key].annotation
        declared = {get_origin(variant) or variant for variant in get_args(annotation)}
        assert expected in declared, f"{key}: validator expects {expected}, structured output declares {annotation}"


def test_build_input_schema_tolerates_missing_graph_sections():
    """A malformed/empty schema must produce unconstrained (enum-free) fields, not crash."""
    schema = build_input_schema({})
    assert "enum" not in _variant(schema["$defs"]["_Resource"]["properties"]["type"], "string")
    assert "enum" not in _variant(schema["$defs"]["_Action"]["properties"]["name"], "string")
    assert "enum" not in _variant(schema["$defs"]["_Subject"]["properties"]["type"], "string")


def test_build_input_schema_ignores_non_string_relations():
    """A malformed relation (a number, an empty string) must not land in the action-name
    enum - a string enum with a non-string member is an invalid schema for Gemini, and the
    post-assembly check (subject_binding._edge_relations) applies the same rule.
    """
    graph_schema = _load_schema_fixture()
    graph_schema["graph"]["edges"] = [{"relation": 123}, {"relation": ""}, {"relation": "RENT"}]
    schema = build_input_schema(graph_schema)
    assert _variant(schema["$defs"]["_Action"]["properties"]["name"], "string")["enum"] == ["RENT"]


@pytest.mark.parametrize(
    ("policies", "expected_enum"),
    [
        pytest.param(
            [
                {"policy": {"actions": ["APPROVE_REFUND"], "resource": {"type": "Car"}}},
                {"policy": {"actions": ["APPROVE_REFUND", "ESCALATE"], "resource": {"type": "Company"}}},
            ],
            ["APPROVE_REFUND", "ESCALATE", "OWN", "RENT", "WORKS_AT"],
            id="widens_with_every_fetched_policys_actions",
        ),
        pytest.param(
            [
                {"policy": {"resource": {"type": "Car"}}},
                {"policy": {"actions": [123, "", "ESCALATE"], "resource": {"type": "Car"}}},
                {"not_a_policy_key": True},
            ],
            ["ESCALATE", "OWN", "RENT", "WORKS_AT"],
            id="ignores_malformed_policy_actions_without_crashing",
        ),
        pytest.param([], ["OWN", "RENT", "WORKS_AT"], id="empty_policies_matches_edges_only"),
    ],
)
def test_build_input_schema_action_names_widened_by_policies(policies, expected_enum):
    """action.name's enum is the union of the schema's edge relations and every fetched
    policy's actions (ENG-9671) - a policy can supply an action outside the schema's own
    vocabulary, gated post-assembly (subject_binding.py) by the action_derivation_rule marker,
    not by this structural enum (JSON Schema can't make an enum conditional per-evaluation). A
    policy missing its actions list, or one with non-string entries, must not crash the enum
    build or leak a non-string member into the string enum; an empty policies list must produce
    exactly the schema-edges-only enum (TDD 4.6's original behavior) - policies are additive,
    never required.
    """
    schema = build_input_schema(_load_schema_fixture(), policies)
    action_name = schema["$defs"]["_Action"]["properties"]["name"]
    assert _variant(action_name, "string")["enum"] == expected_enum


def test_build_input_schema_drops_pydantic_only_keys():
    """`title`/`default` are pydantic-generated noise; `default` isn't even in Gemini's
    supported `response_json_schema` keys (google.genai.types.GenerateContentConfig's
    docstring), so both are stripped from the schema handed to Gemini.
    """
    serialized = json.dumps(build_input_schema(_load_schema_fixture()))
    assert '"title"' not in serialized
    assert '"default"' not in serialized
