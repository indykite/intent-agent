import pytest

from indykite.intent_agent.cel import MAX_FILTER_LENGTH, conjuncts, filter_errors


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("resource.current_price < 100", id="comparison"),
        pytest.param("resource.name.contains('refund') && resource.balance > 1000", id="string_method_and_conjunction"),
        pytest.param("resource.status in ['open', 'pending']", id="membership"),
        pytest.param("resource.create_time > timestamp('2024-01-01T00:00:00Z')", id="timestamp_function"),
        pytest.param("resource.name.matches('^CanBank') || !resource.archived", id="regex_and_negation"),
        pytest.param("resource.tags.exists(t, t == 'policy')", id="comprehension_binds_its_loop_variable"),
        pytest.param("has(resource.description) && size(resource.holdings) > 0", id="has_and_size"),
        pytest.param("double(resource.total) < 9.5 && int(resource.rank) > 2", id="conversion_functions"),
        pytest.param("resource.tags.size() > 0 && resource.name.lowerAscii() == 'x'", id="zero_argument_methods"),
        pytest.param(
            "resource.tags.exists(t, resource.labels.all(u, t == u)) && resource.tags.exists(t, t.size() > 0)",
            id="nested_and_repeated_comprehensions_scope_their_variables",
        ),
    ],
)
def test_filter_errors_accepts_cel_over_the_resource(expression):
    """The skill's CEL subset over `resource` is accepted; a comprehension's loop variable is not foreign."""
    assert filter_errors(expression) == []


def test_filter_errors_reports_syntax_errors_with_a_position():
    """Non-CEL is rejected with the parser's position, so the LLM feedback says where it broke."""
    errors = filter_errors("resource.name contains 'refund'")
    assert len(errors) == 1
    assert errors[0].startswith("context.filters is not valid CEL (line 1, column ")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        pytest.param(
            "request.location == 'Oslo'",
            ["context.filters references 'request'; only resource.<property> may be referenced"],
            id="foreign_root",
        ),
        pytest.param(
            "resource.owner == subject.id",
            ["context.filters references 'subject'; only resource.<property> may be referenced"],
            id="subject_root",
        ),
        pytest.param(
            "dyn(resource.name) == type(resource.id)",
            [
                "context.filters calls dyn(), which is not allowed",
                "context.filters calls type(), which is not allowed",
            ],
            id="reflection_functions",
        ),
        pytest.param(
            ".subject.id == resource.owner",
            ["context.filters references 'subject'; only resource.<property> may be referenced"],
            id="leading_dot_absolute_name",
        ),
        pytest.param(
            ".dyn(resource.name) == 1",
            ["context.filters calls dyn(), which is not allowed"],
            id="leading_dot_global_call",
        ),
        pytest.param(
            "resource.type(1) == 1 || resource.name.dyn() == 'x'",
            [
                "context.filters calls type(), which is not allowed",
                "context.filters calls dyn(), which is not allowed",
            ],
            id="receiver_style_reflection_calls",
        ),
        pytest.param(
            "resource.tags.exists(t, t == 'x') && t == 'y'",
            ["context.filters references 't'; only resource.<property> may be referenced"],
            id="loop_variable_does_not_leak_out_of_its_macro",
        ),
        pytest.param(
            "resource.tags.exists(subject.id, true) && resource.owner == subject.id",
            ["context.filters references 'subject'; only resource.<property> may be referenced"],
            id="macro_argument_cannot_smuggle_a_foreign_variable",
        ),
    ],
)
def test_filter_errors_rejects_foreign_variables_and_blocked_functions(expression, expected):
    """Every non-resource variable and blocked reflection call is reported, each once, all together."""
    assert filter_errors(expression) == expected


def test_filter_errors_rejects_functions_cel_does_not_have():
    """The parser accepts any call name; a non-CEL one would only fail when IBAC evaluates it."""
    assert filter_errors("float(resource.total) > 9.5 || toNumber(resource.rank) > 2") == [
        "context.filters calls float(), not a CEL function (CEL has double(), int(), string(), ...)",
        "context.filters calls toNumber(), not a CEL function (CEL has double(), int(), string(), ...)",
    ]


def test_filter_errors_caps_the_length():
    """A filter past the size cap is rejected before parsing - runaway output, not intent."""
    expression = " || ".join(["resource.name == 'x'"] * (MAX_FILTER_LENGTH // 10))
    assert len(expression) > MAX_FILTER_LENGTH
    assert filter_errors(expression) == [f"context.filters is longer than {MAX_FILTER_LENGTH} characters"]


_CAR_PROPERTIES = {"external_id", "make", "model", "vin", "year"}


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("resource.model == 'V70'", id="dot_access"),
        pytest.param("resource['model'] == 'V70'", id="single_quoted_index"),
        pytest.param('resource["model"] == "V70"', id="double_quoted_index"),
        pytest.param("resource.id == 'kitt' && resource.type == 'Car'", id="authzen_resource_fields_always_known"),
        pytest.param("resource.tags.exists(t, t == 'x')", id="unknown_property_without_a_property_list"),
    ],
)
def test_filter_errors_accepts_known_properties(expression):
    """Known properties, `id` and `type` are accepted; the `tags` case passes no list, which disables the check."""
    properties = None if "tags" in expression else _CAR_PROPERTIES
    assert filter_errors(expression, properties) == []


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        pytest.param(
            "resource.email == 'Volvo'",
            ["context.filters reads resource.email, which is not a property of the resource type"],
            id="dot_access",
        ),
        pytest.param(
            "resource['email'] == 'Volvo'",
            ["context.filters reads resource.email, which is not a property of the resource type"],
            id="index_is_the_same_selection_as_the_dot_form",
        ),
        pytest.param(
            "resource.email == 'a' || resource.colour == 'b'",
            [
                "context.filters reads resource.email, which is not a property of the resource type",
                "context.filters reads resource.colour, which is not a property of the resource type",
            ],
            id="every_unknown_property_at_once",
        ),
        pytest.param(
            "resource['ema' + 'il'] == 'Volvo'",
            [
                (
                    "context.filters indexes resource with an expression instead of a property name, "
                    "so what it reads can't be checked against the resource type"
                )
            ],
            id="computed_key_cannot_be_checked",
        ),
    ],
)
def test_filter_errors_rejects_properties_the_resource_type_lacks(expression, expected):
    """Unknown properties are rejected in either spelling; a computed index can't be checked, so it is refused."""
    assert filter_errors(expression, _CAR_PROPERTIES) == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        pytest.param("resource.a == 'x'", ["resource.a == 'x'"], id="single_term"),
        pytest.param(
            "resource.a == 'x'&&size(resource.b) > 2", ["resource.a == 'x'", "size(resource.b) > 2"], id="call"
        ),
        pytest.param(
            "(resource.a || resource.b) && resource.c", ["(resource.a || resource.b)", "resource.c"], id="group"
        ),
        pytest.param("resource.a || resource.b && resource.c", ["resource.a || resource.b && resource.c"], id="top_or"),
        pytest.param("resource.a == 'x && y'", ["resource.a == 'x && y'"], id="and_in_literal"),
        pytest.param("resource.a contains 'x'", None, id="not_cel"),
    ],
)
def test_conjuncts_splits_only_the_top_level_and_chain(expression, expected):
    assert conjuncts(expression) == expected
