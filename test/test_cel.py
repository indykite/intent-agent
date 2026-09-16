import pytest

from indykite.intent_agent.cel import MAX_FILTER_LENGTH, filter_errors


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
        pytest.param("resource.tags.size() > 0 && resource.name.lowerAscii() == 'x'", id="zero_argument_methods"),
        pytest.param(
            "resource.tags.exists(t, resource.labels.all(u, t == u)) && resource.tags.exists(t, t.size() > 0)",
            id="nested_and_repeated_comprehensions_scope_their_variables",
        ),
    ],
)
def test_filter_errors_accepts_cel_over_the_resource(expression):
    """The skill's CEL subset - comparisons, string methods, membership, timestamps, macros,
    `has`/`size` - over `resource` is accepted as-is; a comprehension's loop variable is not a
    foreign reference.
    """
    assert filter_errors(expression) == []


def test_filter_errors_reports_syntax_errors_with_a_position():
    """Non-CEL (the SQL / pseudo-grammar the model used to produce) is rejected with the
    parser's position, so the feedback tells the LLM where the expression broke.
    """
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
    """A filter constrains the resource only: any other variable - in plain or leading-dot
    form, outside a comprehension's own body, or smuggled in as a macro argument - and the
    reflection functions the platform's evaluator also blocks, called globally or on a
    receiver, are reported - each once, all at the same time.
    """
    assert filter_errors(expression) == expected


def test_filter_errors_caps_the_length():
    """A filter past the size cap is rejected before parsing - runaway output, not intent."""
    expression = " || ".join(["resource.name == 'x'"] * (MAX_FILTER_LENGTH // 10))
    assert len(expression) > MAX_FILTER_LENGTH
    assert filter_errors(expression) == [f"context.filters is longer than {MAX_FILTER_LENGTH} characters"]
