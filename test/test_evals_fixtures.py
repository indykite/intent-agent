"""Golden fixture loading (evals/fixtures.py)."""

import json

import pytest

from evals.fixtures import (
    ADVERSARIAL_SUBJECT_SPOOF,
    AMBIGUOUS,
    CATEGORIES,
    OUT_OF_SCOPE,
    SINGLE_INTENT,
    SUBJECT_SEARCH,
    FixtureError,
    canonical_category,
    digital_twin_types,
    load_fixture,
    load_fixtures,
    load_schema,
)
from indykite.intent_agent.errors import no_intent_error


def test_loads_an_evaluations_fixture(eval_fixture_file):
    fixture = load_fixture(eval_fixture_file("single_intent_test_1"))

    assert fixture.id == "single_intent_test_1"
    assert fixture.prompt == "Can I access the document doc123?"
    assert fixture.jwt_sub == "leslie"
    assert fixture.subject_type == "User"  # inferred: user_model's only DigitalTwin type
    assert fixture.category == SINGLE_INTENT
    assert fixture.data_model == "user_model"
    assert fixture.schema_path.name == "user_model_schema.json"
    assert fixture.expected["evaluations"][0]["action"] == {"name": "CAN_ACCESS"}
    assert not fixture.expects_refusal


def test_loads_a_refusal_fixture(eval_fixture_file):
    prompt = "Is my driving licence still valid?"
    path = eval_fixture_file(
        "out_of_scope_test_1", prompt=prompt, category="out of scope", expected=no_intent_error(prompt)
    )

    fixture = load_fixture(path)

    assert fixture.expects_refusal
    assert fixture.category == OUT_OF_SCOPE
    assert fixture.expected["reason"] == no_intent_error(prompt)["reason"]


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [
        ("single-intent", SINGLE_INTENT),
        ("Subject Search", SUBJECT_SEARCH),
        ("adversarial_subject_spoof", ADVERSARIAL_SUBJECT_SPOOF),
        ("  out   of scope ", OUT_OF_SCOPE),
        ("Adversarial subject-spoof", ADVERSARIAL_SUBJECT_SPOOF),
    ],
)
def test_category_spellings_are_canonicalised(spelling, canonical):
    """Category spellings that differ in case, separators and spacing map to one canonical name."""
    assert canonical_category(spelling) == canonical


def test_unknown_category_is_rejected_naming_the_file(eval_fixture_file):
    path = eval_fixture_file("typo_test_1", category="adversarial subject spooof")

    with pytest.raises(FixtureError, match=r"typo_test_1\.json: unknown category 'adversarial subject spooof'"):
        load_fixture(path)


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"prompt": ""}, "'prompt' must be a non-empty string"),
        ({"prompt": 42}, "'prompt' must be a non-empty string"),
        ({"jwt_sub": None}, "'jwt_sub' must be a non-empty string"),
        ({"category": ""}, "'category' must be a non-empty string"),
        ({"data_model": ""}, "'data_model' must be a non-empty string"),
        ({"data_model": "missing_model"}, "data model 'missing_model' has no schema file"),
        # data_model is interpolated into file names, so it must be a single path segment.
        ({"data_model": "../outside"}, "data model '../outside' is not a bare model name"),
        ({"data_model": "nested/user_model"}, "data model 'nested/user_model' is not a bare model name"),
        ({"data_model": ".."}, "data model '..' is not a bare model name"),
        ({"expected": "yes"}, "'expected' must be an object"),
        ({"expected": {"prompt": "Can I access the document doc123?", "evaluations": []}}, "non-empty array"),
        ({"expected": {"prompt": "Can I access the document doc123?", "evaluations": ["x"]}}, "non-empty array"),
        ({"expected": {"prompt": "Can I access the document doc123?", "error": "e"}}, "either"),
        (
            {
                "expected": {
                    "prompt": "Can I access the document doc123?",
                    "error": "e",
                    "reason": "r",
                    "obligation": "o",
                    "x": 1,
                }
            },
            "either",
        ),
        (
            {"expected": {"prompt": "something else", "error": "e", "reason": "r", "obligation": "o"}},
            "must echo 'prompt' verbatim",
        ),
        ({"subject_type": " "}, "'subject_type' must be a non-empty string"),
        ({"subject_type": "Ghost"}, "subject_type 'Ghost' is not a node type of data model 'user_model'"),
    ],
)
def test_malformed_fixture_is_rejected(eval_fixture_file, fields, problem):
    path = eval_fixture_file("bad_test_1", **fields)

    with pytest.raises(FixtureError, match=rf"bad_test_1\.json: .*{problem}"):
        load_fixture(path)


@pytest.mark.parametrize(
    ("evaluation", "problem"),
    [
        ({"resource": "Document"}, "'resource' must be an object with a non-empty string 'type'"),
        ({"resource": {"type": " "}}, "'resource' must be an object with a non-empty string 'type'"),
        ({"action": "CAN_ACCESS", "resource": {"type": "Document"}}, "'action', when present, must be an object"),
        ({"action": {}, "resource": {"type": "Document"}}, "'action', when present, must be an object"),
        ({"action": {"name": ""}, "resource": {"type": "Document"}}, "'action', when present, must be an object"),
        ({"subject": "leslie", "resource": {"type": "Document"}}, "'subject', when present, must be an object"),
        ({"resource": {"type": "Document"}, "context": "bad"}, "'context', when present, must be an object"),
        ({"resource": {"type": "Document"}, "context": {"parameters": []}}, "'context.parameters', when present"),
        (
            {"resource": {"type": "Document"}, "context": {"filters": 1}},
            "'context.filters', when present, must be a string",
        ),
    ],
)
def test_golden_evaluation_with_a_shape_the_pipeline_never_returns_is_rejected(eval_fixture_file, evaluation, problem):
    """A golden evaluation the pipeline would reject or drop is a fixture error at load time."""
    expected = {"prompt": "Can I access the document doc123?", "evaluations": [evaluation]}
    path = eval_fixture_file("bad_test_1", expected=expected)

    with pytest.raises(FixtureError, match=rf"bad_test_1\.json: 'expected\.evaluations\[0\]': {problem}"):
        load_fixture(path)


def test_missing_key_is_rejected(eval_fixture_file):
    path = eval_fixture_file("bad_test_1")
    document = json.loads(path.read_text())
    del document["jwt_sub"]
    path.write_text(json.dumps(document))

    with pytest.raises(FixtureError, match=r"'jwt_sub' must be a non-empty string"):
        load_fixture(path)


def test_loads_the_confidence_level_as_written(eval_fixture):
    prompt = "Is my driving license still valid?"
    refusal = eval_fixture(
        "out_of_scope_test_1", prompt=prompt, category=OUT_OF_SCOPE, expected=no_intent_error(prompt)
    )
    rated_refusal = eval_fixture(
        "out_of_scope_test_2", prompt=prompt, category=OUT_OF_SCOPE, expected=no_intent_error(prompt), confidence="high"
    )

    assert eval_fixture(confidence="medium").confidence == "medium"
    assert eval_fixture(confidence="certain").confidence == "certain"
    assert eval_fixture(confidence=None).confidence is None
    assert refusal.confidence is None
    assert rated_refusal.confidence == "high"


def test_blank_jwt_sub_is_an_anonymous_caller(eval_fixture_file):
    """An anonymous caller is bound without an id, so the golden subject carries the type only."""
    expected = {
        "prompt": "Can I access the document doc123?",
        "evaluations": [
            {
                "subject": {"type": "User"},
                "action": {"name": "CAN_ACCESS"},
                "resource": {"type": "Document", "id": "doc123"},
                "context": {"parameters": {}},
            }
        ],
    }

    assert load_fixture(eval_fixture_file(jwt_sub="", expected=expected)).jwt_sub == ""


def test_invalid_json_and_non_object_documents_are_rejected(tmp_path, eval_schema_file):
    eval_schema_file()
    broken = tmp_path / "evals" / "broken_test_1.json"
    broken.write_text('{"prompt": ')
    array = tmp_path / "evals" / "array_test_1.json"
    array.write_text("[]")

    with pytest.raises(FixtureError, match=r"broken_test_1\.json: not valid JSON \(.* at line 1\)"):
        load_fixture(broken)
    with pytest.raises(FixtureError, match=r"array_test_1\.json: the fixture must be a JSON object"):
        load_fixture(array)


_CUSTOMER_EXPECTED = {
    "prompt": "Can I access the document doc123?",
    "evaluations": [
        {
            "subject": {"type": "Customer", "id": "leslie"},
            "action": {"name": "CAN_ACCESS"},
            "resource": {"type": "Document", "id": "doc123"},
            "context": {"parameters": {}},
        }
    ],
}


def test_explicit_subject_type_wins_over_inference(eval_fixture_file, eval_schema_file, eval_policies_file):
    eval_schema_file(
        "combined_model", digital_twins=("User", "Customer"), relations=[("Customer", "CAN_ACCESS", "Document")]
    )
    eval_policies_file("combined_model", [("Customer", "CAN_ACCESS", "Document")])

    fixture = load_fixture(
        eval_fixture_file(data_model="combined_model", subject_type="Customer", expected=_CUSTOMER_EXPECTED)
    )

    assert fixture.subject_type == "Customer"


def test_subject_type_is_required_when_the_model_has_several_digital_twins(eval_fixture_file, eval_schema_file):
    eval_schema_file("combined_model", digital_twins=("User", "Customer"))
    path = eval_fixture_file(data_model="combined_model")

    with pytest.raises(
        FixtureError, match=r"'subject_type' is required - .* has 2 DigitalTwin types \(Customer, User\)"
    ):
        load_fixture(path)


def test_subject_type_is_required_when_the_model_has_no_digital_twin(eval_fixture_file, eval_schema_file):
    """_Application carries the DigitalTwin label too, but is the platform's own node, never a caller."""
    eval_schema_file("bare_model", digital_twins=())
    path = eval_fixture_file(data_model="bare_model")

    assert digital_twin_types(load_schema(path.parent / "schema" / "bare_model_schema.json")) == []
    with pytest.raises(FixtureError, match=r"has 0 DigitalTwin types \(none\)"):
        load_fixture(path)


def test_digital_twin_types_tolerates_a_non_jgf_document():
    assert digital_twin_types({"graph": {"nodes": "?"}}) == []
    assert digital_twin_types({"graph": {"nodes": {"User": "?", "Bot": {"metadata": {"system_labels": ["x"]}}}}}) == []


def test_load_fixtures_returns_every_file_sorted_and_skips_the_schema_dir(tmp_path, eval_fixture_file):
    eval_fixture_file("multi_intent_test_1", category="multi intent")
    eval_fixture_file("adversarial_subject_spoof_test_1", category="adversarial subject spoof")

    fixtures = load_fixtures(tmp_path / "evals")

    assert [fixture.id for fixture in fixtures] == ["adversarial_subject_spoof_test_1", "multi_intent_test_1"]


def test_load_fixtures_accepts_a_separate_schema_dir(tmp_path, eval_fixture_file, eval_schema_file, eval_policies_file):
    elsewhere = tmp_path / "models"
    eval_schema_file(directory=elsewhere)
    eval_policies_file("user_model", [("User", "CAN_ACCESS", "Document")], directory=elsewhere)
    fixtures_dir = tmp_path / "only-fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "single_intent_test_1.json").write_bytes(eval_fixture_file().read_bytes())

    fixtures = load_fixtures(fixtures_dir, schema_dir=elsewhere / "schema")

    assert [fixture.schema_path for fixture in fixtures] == [elsewhere / "schema" / "user_model_schema.json"]


def test_load_fixtures_rejects_an_empty_directory(tmp_path):
    with pytest.raises(FixtureError, match="no fixture files"):
        load_fixtures(tmp_path)


def test_shipped_suite_loads_and_covers_the_gated_categories():
    """Every shipped fixture loads, the gated categories are present, and golden types exist in the schema."""
    fixtures = load_fixtures()

    assert len(fixtures) >= 20  # noqa: PLR2004 - the suite as first committed; only ever grows
    categories = {fixture.category for fixture in fixtures}
    assert categories <= set(CATEGORIES)
    assert categories >= set(CATEGORIES) - {AMBIGUOUS}  # ambiguous fixtures are welcome, not yet required
    assert any(fixture.category == ADVERSARIAL_SUBJECT_SPOOF for fixture in fixtures)
    assert any(fixture.expects_refusal for fixture in fixtures)
    for fixture in fixtures:
        node_types = set(load_schema(fixture.schema_path)["graph"]["nodes"])
        for evaluation in fixture.expected.get("evaluations", []):
            assert evaluation["resource"]["type"] in node_types, (fixture.id, evaluation)
            assert evaluation.get("subject", {}).get("type", fixture.subject_type) in node_types, (
                fixture.id,
                evaluation,
            )


def test_policies_file_next_to_the_schema_is_picked_up_and_widens_the_action_vocabulary(
    eval_fixture_file, eval_schema_file, eval_policies_file
):
    eval_schema_file()
    eval_policies_file("user_model", [("User", "CHECK_MEMBERSHIP", "Document")])
    expected = {
        "prompt": "Can I access the document doc123?",
        "evaluations": [
            {
                "subject": {"type": "User", "id": "leslie"},
                "action": {"name": "CHECK_MEMBERSHIP"},
                "resource": {"type": "Document"},
                "context": {"parameters": {}},
            }
        ],
    }

    fixture = load_fixture(eval_fixture_file(expected=expected))

    assert fixture.policies_path == fixture.schema_path.with_name("user_model_policies.json")
    assert fixture.policies[0]["policy"]["actions"] == ["CHECK_MEMBERSHIP"]


def test_golden_answer_must_carry_the_parameters_object_the_pipeline_sets(
    eval_fixture_file, eval_schema_file, eval_policies_file
):
    """A golden answer without the `context.parameters` object the pipeline always sets is rejected."""
    eval_schema_file()
    eval_policies_file("user_model", [("User", "CHECK_MEMBERSHIP", "Document")])
    expected = {
        "prompt": "Can I access the document doc123?",
        "evaluations": [{"action": {"name": "CHECK_MEMBERSHIP"}, "resource": {"type": "Document"}, "context": {}}],
    }

    with pytest.raises(
        FixtureError, match=r"returns a different response .* evaluations\[0\] would come back as .*parameters"
    ):
        load_fixture(eval_fixture_file(expected=expected))


@pytest.mark.parametrize(
    ("evaluations", "problem"),
    [
        pytest.param(
            [
                {
                    "subject": {"type": "User", "id": "roy"},
                    "action": {"name": "CAN_ACCESS"},
                    "resource": {"type": "Document"},
                    "context": {},
                }
            ],
            r"evaluations\[0\] would come back as .*\"id\": \"leslie\"",
            id="second_subject_at_index_0_is_rebound_to_the_caller",
        ),
        pytest.param(
            [
                {
                    "subject": {"type": "User", "id": "leslie"},
                    "action": {"name": "CAN_ACCESS"},
                    "resource": {"type": "Document"},
                    "context": {"parameters": {}},
                },
                {
                    "subject": {"type": "User", "id": "leslie"},
                    "action": {"name": "CAN_ACCESS"},
                    "resource": {"type": "Document"},
                    "context": {"parameters": {}},
                },
            ],
            r"evaluations\[1\] would not come back at all",
            id="literal_duplicate_is_dropped",
        ),
        pytest.param(
            [
                {
                    "subject": {"type": "Document"},
                    "action": {"name": "CAN_ACCESS"},
                    "resource": {"type": "Document"},
                    "context": {},
                }
            ],
            r"the response would be a refusal",
            id="subject_search_over_a_resource_only_type_is_dropped",
        ),
    ],
)
def test_golden_answer_the_pipeline_rewrites_is_rejected(eval_fixture_file, evaluations, problem):
    """A golden answer that normalisation would rebind, drop or refuse is a fixture error."""
    expected = {"prompt": "Can I access the document doc123?", "evaluations": evaluations}

    with pytest.raises(FixtureError, match=rf"the pipeline returns a different response .* {problem}"):
        load_fixture(eval_fixture_file(expected=expected))


def test_fixture_without_a_policies_file_has_the_empty_listing(eval_fixture_file):
    """Only a refusal fixture loads without a policies file."""
    path = eval_fixture_file(expected=no_intent_error("Can I access the document doc123?"))
    (path.parent / "schema" / "user_model_policies.json").unlink()

    fixture = load_fixture(path)

    assert fixture.policies_path is None
    assert fixture.policies == []


@pytest.mark.parametrize(
    ("triple", "reason"),
    [
        (("Customer", "CHECK_MEMBERSHIP", "Document"), "another subject type"),
        (("User", "CHECK_MEMBERSHIP", "Folder"), "another resource type"),
    ],
)
def test_golden_policy_action_is_checked_for_the_caller_and_the_resource_type(
    eval_fixture_file, eval_schema_file, eval_policies_file, triple, reason
):
    """A policy for another subject or resource type doesn't make the golden action producible."""
    eval_schema_file(digital_twins=("User", "Customer"), resources=("Document", "Folder"))
    eval_policies_file("user_model", [triple])
    expected = {
        "prompt": "Can I access the document doc123?",
        "evaluations": [{"action": {"name": "CHECK_MEMBERSHIP"}, "resource": {"type": "Document"}, "context": {}}],
    }
    path = eval_fixture_file(expected=expected, subject_type="User")

    with pytest.raises(FixtureError, match=r"grants subject type 'User' on resource type 'Document'"):
        load_fixture(path)


_CALLER = {"type": "User", "id": "leslie"}


_MARKED = {"parameters": {}}
_SECOND_HOP = {"parameters": {}}
_FOLDER_MODEL = {
    "resources": ("Folder", "Document"),
    "relations": [("User", "CAN_ACCESS", "Folder"), ("Folder", "CONTAINS", "Document")],
}
_FOLDER_POLICIES = [("User", "CAN_ACCESS", "Folder"), ("User", "CAN_ACCESS", "Document")]


def _hop(action, resource_type, context):
    return {"subject": _CALLER, "action": {"name": action}, "resource": {"type": resource_type}, "context": context}


def _search(resource_type):
    return {
        "subject": {"type": "User"},
        "action": {"name": "CAN_ACCESS"},
        "resource": {"type": resource_type, "id": "doc-1"},
        "context": _MARKED,
    }


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            (
                None,
                None,
                [{"action": {"name": "CHECK_MEMBERSHIP"}, "resource": {"type": "Document"}, "context": {}}],
                r"evaluations\[0\] names action 'CHECK_MEMBERSHIP', which no policy in user_model_policies\.json",
            ),
            id="action_no_policy_grants",
        ),
        pytest.param(
            (
                {"resources": ("Document", "Account"), "relations": [("User", "OWNS", "Account")]},
                None,
                [{"subject": _CALLER, "action": {"name": "OWNS"}, "resource": {"type": "Account"}}],
                r"names action 'OWNS', which no policy .* on resource type 'Account'",
            ),
            id="callers_own_edge_is_no_action",
        ),
        pytest.param(({"resources": ("Document", "Account")}, None, [_search("Document")], None), id="granted_search"),
        pytest.param(
            (
                {"resources": ("Document", "Account")},
                None,
                [_search("Account")],
                r"names action 'CAN_ACCESS', which no policy",
            ),
            id="search_no_policy_grants",
        ),
        pytest.param(
            (
                _FOLDER_MODEL,
                _FOLDER_POLICIES,
                [_hop("CAN_ACCESS", "Folder", _MARKED), _hop("CAN_ACCESS", "Document", _SECOND_HOP)],
                None,
            ),
            id="second_hop_with_its_own_policy",
        ),
        pytest.param(
            (
                _FOLDER_MODEL,
                _FOLDER_POLICIES,
                [_hop("CAN_ACCESS", "Folder", _MARKED), _hop("CONTAINS", "Document", _SECOND_HOP)],
                r"evaluations\[1\] names action 'CONTAINS', which no policy",
            ),
            id="second_hop_over_an_edge",
        ),
    ],
)
def test_golden_answer_loads_only_as_a_policy_combination(
    eval_fixture_file, eval_schema_file, eval_policies_file, case
):
    model, policies, evaluations, error = case
    if model is not None:
        eval_schema_file(**model)
    if policies is not None:
        eval_policies_file("user_model", policies)
    expected = {"prompt": "Can I access the document doc123?", "evaluations": evaluations}
    path = eval_fixture_file(expected=expected)

    if error is None:
        assert load_fixture(path).expected == expected
    else:
        with pytest.raises(FixtureError, match=error):
            load_fixture(path)


def test_shipped_suite_has_a_policies_file_for_every_data_model_it_uses():
    """Every golden answer needs its data model's policies to be producible; a refusal may have none."""
    for fixture in load_fixtures():
        assert fixture.policies_path is not None, fixture.id
        assert fixture.policies or fixture.expects_refusal, fixture.id


def test_every_adversarial_category_feeds_the_injection_gate():
    from evals.fixtures import ADVERSARIAL_OVERSIZED_INPUT, ADVERSARIAL_SQL_INJECTION, is_adversarial

    assert is_adversarial(ADVERSARIAL_SUBJECT_SPOOF)
    assert is_adversarial(ADVERSARIAL_SQL_INJECTION)
    assert is_adversarial(ADVERSARIAL_OVERSIZED_INPUT)
    assert canonical_category("Adversarial SQL-injection") == ADVERSARIAL_SQL_INJECTION
    assert canonical_category("adversarial_oversized_input") == ADVERSARIAL_OVERSIZED_INPUT
    assert not is_adversarial(OUT_OF_SCOPE)


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        ('{"graph": ', "data model schema is not valid JSON"),
        ("[]", "data model schema must be a JSON object"),
        ('{"nodes": {}}', "must have a 'graph' object"),
        ('{"graph": {"nodes": [], "edges": []}}', "must have a 'graph' object"),
    ],
)
def test_malformed_schema_file_is_a_fixture_error(eval_fixture_file, content, problem):
    """A malformed data-model file raises FixtureError, not a raw json or attribute error."""
    path = eval_fixture_file()
    (path.parent / "schema" / "user_model_schema.json").write_text(content, encoding="utf-8")
    load_schema.cache_clear()

    with pytest.raises(FixtureError, match=rf"user_model_schema\.json: .*{problem}"):
        load_fixture(path)


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        ("{", "policy listing is not valid JSON"),
        ('"nope"', "policy listing must be a JSON object"),
        ('{"policies": []}', "must have a 'results' array"),
        ('{"results": [{"tags": []}]}', "must have a 'results' array"),
        ('{"results": ["x"]}', "must have a 'results' array"),
    ],
)
def test_malformed_policies_file_is_a_fixture_error(eval_fixture_file, content, problem):
    path = eval_fixture_file()
    (path.parent / "schema" / "user_model_policies.json").write_text(content, encoding="utf-8")
    from evals.fixtures import load_policies

    load_policies.cache_clear()

    with pytest.raises(FixtureError, match=rf"user_model_policies\.json: .*{problem}"):
        load_fixture(path)
