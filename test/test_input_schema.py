import json
from pathlib import Path
from typing import Any

from indykite.intent_agent.input_schema import build_input_schema

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
    """subject.type must enumerate every node label, same as resource.type - the LLM decides
    subject.type itself now (ENG-9668), not restricted to any particular labelled subset.
    """
    schema = build_input_schema(_load_schema_fixture())
    subject_type = schema["$defs"]["_Subject"]["properties"]["type"]
    assert _variant(subject_type, "string")["enum"] == ["Car", "Company", "Customer", "_Application"]


def test_build_input_schema_omits_subject_id_entirely():
    """subject.id must never be part of the LLM's output schema - it's always assigned
    deterministically post-LLM from the validated access token (TDD 4.7), never LLM-supplied.
    """
    schema = build_input_schema(_load_schema_fixture())
    assert "id" not in schema["$defs"]["_Subject"]["properties"]


def test_build_input_schema_context_has_exactly_the_tdd_keys():
    """context must expose exactly parameters/filters/qualifiers (TDD 4.6), not an open object."""
    schema = build_input_schema(_load_schema_fixture())
    context_properties = schema["$defs"]["_Context"]["properties"]
    assert set(context_properties.keys()) == {"parameters", "filters", "qualifiers"}
    _variant(context_properties["filters"], "string")
    _variant(context_properties["parameters"], "object")
    _variant(context_properties["qualifiers"], "object")


def test_build_input_schema_top_level_requires_prompt_and_evaluations():
    """The LLM's output shape is deliberately just {prompt, evaluations} - no error shape;
    our own code is solely responsible for building an error response (subject_binding.py).
    """
    schema = build_input_schema(_load_schema_fixture())
    assert schema["required"] == ["prompt", "evaluations"]
    assert set(schema["properties"].keys()) == {"prompt", "evaluations"}


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


def test_build_input_schema_drops_pydantic_only_keys():
    """`title`/`default` are pydantic-generated noise; `default` isn't even in Gemini's
    supported `response_json_schema` keys (google.genai.types.GenerateContentConfig's
    docstring), so both are stripped from the schema handed to Gemini.
    """
    serialized = json.dumps(build_input_schema(_load_schema_fixture()))
    assert '"title"' not in serialized
    assert '"default"' not in serialized
