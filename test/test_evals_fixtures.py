"""Golden fixture loading (evals/fixtures.py): the shape every fixture file must have, the
category spellings, the caller's subject type, and the shipped suite itself.
"""

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
    """The TDD, the fixtures and the task tracker each spell the categories differently."""
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
    """A golden evaluation the pipeline would reject or drop (a malformed resource/action/context,
    a blank action name) can never be matched, so it is a fixture error at load time - not a
    traceback out of the producibility check, and not a run that reads as a model failure.
    """
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


def test_blank_jwt_sub_is_an_anonymous_caller(eval_fixture_file):
    """An anonymous caller is bound without an id, so the golden subject carries the type only."""
    expected = {
        "prompt": "Can I access the document doc123?",
        "evaluations": [
            {
                "subject": {"type": "User"},
                "action": {"name": "CAN_ACCESS"},
                "resource": {"type": "Document", "id": "doc123"},
                "context": {},
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
            "context": {},
        }
    ],
}


def test_explicit_subject_type_wins_over_inference(eval_fixture_file, eval_schema_file):
    eval_schema_file(
        "combined_model", digital_twins=("User", "Customer"), relations=[("Customer", "CAN_ACCESS", "Document")]
    )

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


def test_load_fixtures_accepts_a_separate_schema_dir(tmp_path, eval_fixture_file, eval_schema_file):
    elsewhere = tmp_path / "models"
    eval_schema_file(directory=elsewhere)
    fixtures_dir = tmp_path / "only-fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "single_intent_test_1.json").write_bytes(eval_fixture_file().read_bytes())

    fixtures = load_fixtures(fixtures_dir, schema_dir=elsewhere / "schema")

    assert [fixture.schema_path for fixture in fixtures] == [elsewhere / "schema" / "user_model_schema.json"]


def test_load_fixtures_rejects_an_empty_directory(tmp_path):
    with pytest.raises(FixtureError, match="no fixture files"):
        load_fixtures(tmp_path)


def test_shipped_suite_loads_and_covers_the_gated_categories():
    """The fixtures in evals/ are the validated spec CI runs the pipeline against: every one must
    load, the injection gate needs adversarial fixtures to exist, the forced-match gate needs
    refusal fixtures, and every subject/resource type a golden answer names must be a node type
    of its own data model (a caller or resource type the schema doesn't have can't be bound).
    """
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
                "context": {"parameters": {"action_derivation_rule": "policy_based"}},
            }
        ],
    }

    fixture = load_fixture(eval_fixture_file(expected=expected))

    assert fixture.policies_path == fixture.schema_path.with_name("user_model_policies.json")
    assert fixture.policies[0]["policy"]["actions"] == ["CHECK_MEMBERSHIP"]


def test_golden_policy_action_must_carry_the_marker_the_pipeline_sets(
    eval_fixture_file, eval_schema_file, eval_policies_file
):
    """Normalisation completes `action_derivation_rule: policy_based` on a policy-granted action
    by itself, so a golden answer without it could never equal the response: rejected at load.
    """
    eval_schema_file()
    eval_policies_file("user_model", [("User", "CHECK_MEMBERSHIP", "Document")])
    expected = {
        "prompt": "Can I access the document doc123?",
        "evaluations": [{"action": {"name": "CHECK_MEMBERSHIP"}, "resource": {"type": "Document"}, "context": {}}],
    }

    with pytest.raises(
        FixtureError, match=r"returns a different response .* evaluations\[0\] would come back as .*policy_based"
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
                    "context": {},
                },
                {
                    "subject": {"type": "User", "id": "leslie"},
                    "action": {"name": "CAN_ACCESS"},
                    "resource": {"type": "Document"},
                    "context": {},
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
    """The loader feeds the golden answer through the real normalisation: whatever the pipeline
    would rebind, drop or refuse can never come back as written, so it is a fixture error.
    """
    expected = {"prompt": "Can I access the document doc123?", "evaluations": evaluations}

    with pytest.raises(FixtureError, match=rf"the pipeline returns a different response .* {problem}"):
        load_fixture(eval_fixture_file(expected=expected))


def test_fixture_without_a_policies_file_has_the_empty_listing(eval_fixture):
    fixture = eval_fixture()

    assert fixture.policies_path is None
    assert fixture.policies == []


def test_golden_action_that_no_edge_or_policy_provides_is_rejected(eval_fixture_file):
    expected = {
        "prompt": "Can I access the document doc123?",
        "evaluations": [{"action": {"name": "CHECK_MEMBERSHIP"}, "resource": {"type": "Document"}, "context": {}}],
    }
    path = eval_fixture_file("unreachable_test_1", expected=expected)

    with pytest.raises(
        FixtureError, match=r"evaluations\[0\] names action 'CHECK_MEMBERSHIP', which is neither an edge"
    ):
        load_fixture(path)


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
    """The loader's check is scoped the way the run is: the runner serves only the caller's
    subject type's policies and the pipeline accepts a policy action only on its own resource
    type, so a policy for {reason} doesn't make the golden action producible.
    """
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


def test_golden_edge_action_must_be_the_callers_own_relationship(eval_fixture_file, eval_schema_file):
    """An edge label that exists in the data model, but for another pair of types, doesn't make
    the caller-bound evaluation producible: the pipeline's relationship check rejects it at run
    time, so the loader does too, with the same reason.
    """
    eval_schema_file(resources=("Folder", "Account"), relations=[("User", "CAN_ACCESS", "Folder")])
    expected = {
        "prompt": "Can I access the document doc123?",
        "evaluations": [{"subject": _CALLER, "action": {"name": "CAN_ACCESS"}, "resource": {"type": "Account"}}],
    }

    with pytest.raises(FixtureError, match=r"evaluations\[0\]: action.name 'CAN_ACCESS' for subject type 'User'"):
        load_fixture(eval_fixture_file(expected=expected))


def test_golden_subject_search_is_exempt_from_the_callers_relationship_check(eval_fixture_file, eval_schema_file):
    """A Subject Search (a bare subject type, no id) isn't bound to the caller, so the same
    action on the same resource type is producible there - the loader reads the golden shape the
    way the binding produced it.
    """
    eval_schema_file(resources=("Folder", "Account"), relations=[("User", "CAN_ACCESS", "Folder")])
    expected = {
        "prompt": "Can I access the document doc123?",
        "evaluations": [
            {
                "subject": {"type": "User"},
                "action": {"name": "CAN_ACCESS"},
                "resource": {"type": "Account"},
                "context": {},
            }
        ],
    }

    assert load_fixture(eval_fixture_file(expected=expected)).expected == expected


def test_golden_dependent_hop_is_grounded_through_the_previous_hop(eval_fixture_file, eval_schema_file):
    """A chain's later hop may use the edge from the previous hop's type (Folder CONTAINS
    Document) - the pipeline's dependent-hop rule - so it loads although User holds no such edge.
    """
    eval_schema_file(
        resources=("Folder", "Document"),
        relations=[("User", "CAN_ACCESS", "Folder"), ("Folder", "CONTAINS", "Document")],
    )
    expected = {
        "prompt": "Can I access the document doc123?",
        "evaluations": [
            {"subject": _CALLER, "action": {"name": "CAN_ACCESS"}, "resource": {"type": "Folder"}, "context": {}},
            {
                "subject": _CALLER,
                "action": {"name": "CONTAINS"},
                "resource": {"type": "Document"},
                "context": {
                    "parameters": {"depends_on_previous_evaluations": True, "action_derivation_rule": "policy_based"}
                },
            },
        ],
    }

    fixture = load_fixture(eval_fixture_file(expected=expected))

    assert [evaluation["action"]["name"] for evaluation in fixture.expected["evaluations"]] == [
        "CAN_ACCESS",
        "CONTAINS",
    ]


def test_shipped_suite_has_a_policies_file_for_every_data_model_it_uses():
    """The policy fixtures are what make the six policy-action golden answers producible; a
    data model without one would fail the loader's producibility check first.
    """
    for fixture in load_fixtures():
        assert fixture.policies_path is not None, fixture.id
        assert fixture.policies, fixture.id


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
    """A bad data-model file follows the same error path as a bad fixture (exit 2 from the CLI),
    never a traceback out of json or a later AttributeError.
    """
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
