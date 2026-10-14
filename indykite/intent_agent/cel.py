"""CEL enforcement for `context.filters` (PRD context semantics, AuthZEN/COAZ precedent).

A filter is the one place the extraction expresses a condition, and the contract is Common
Expression Language (https://cel.dev/) over the resource's own properties. This module checks a
filter the way the platform's `jarvis/pkg/cel` guards the scripts it runs - parse it, allow
one variable, block the reflection functions, cap the size - but never evaluates it: that is
the IBAC Engine's job downstream. A failed check is a validation error like an off-schema
type: fed back to the LLM through the retry loop (errors.validation_feedback_error), not
returned to the caller.
"""

from dataclasses import dataclass, field

import celpy
from lark import Tree

FILTER_VARIABLE = "resource"
# Like the platform's evaluator: no host-type reflection from a filter, whether called as
# `dyn(x)` or receiver-style as `x.dyn()`.
_BLOCKED_FUNCTIONS = frozenset({"dyn", "type"})
# CEL's standard global functions and the `has` macro. The parser accepts any name, so the model's
# `float(x)` / `toFloat(x)` (no such functions) would only fail when IBAC evaluates the filter.
_GLOBAL_FUNCTIONS = frozenset(
    {"bool", "bytes", "double", "duration", "has", "int", "matches", "size", "string", "timestamp", "uint"}
)
# Comprehension macros bind their first argument as a loop variable, a legitimate identifier
# inside the macro's remaining arguments only (`resource.tags.exists(t, t == 'x')`).
_COMPREHENSION_MACROS = frozenset({"all", "exists", "exists_one", "filter", "map"})
# Generous for a prompt-derived condition; a filter past it is runaway output, not intent.
MAX_FILTER_LENGTH = 1000
# The opening and closing quote of a string literal, the shortest form `_string_literal` accepts.
_QUOTE_PAIR = 2

# celpy's grammar rule names: a variable reference, with or without CEL's leading-dot absolute
# form (`.subject.id`), and a global function call in the same two spellings.
_VARIABLE_NODES = frozenset({"ident", "dot_ident"})
_GLOBAL_CALL_NODES = frozenset({"ident_arg", "dot_ident_arg"})
_METHOD_CALL_NODE = "member_dot_arg"
# CEL selects a field two ways, `resource.email` and `resource["email"]`, and they mean the same
# thing - so the property check has to see both or the bracket form walks past it.
_MEMBER_DOT_NODE = "member_dot"
_MEMBER_INDEX_NODE = "member_index"
_MEMBER_ACCESS_NODES = frozenset({_MEMBER_DOT_NODE, _MEMBER_INDEX_NODE})

# Fields of the AuthZEN resource object itself rather than of the graph node, so a filter may
# name them even though the data schema's property list never does.
_ALWAYS_KNOWN_PROPERTIES = frozenset({"id", "type"})

_environment = celpy.Environment()


@dataclass
class _Findings:
    """What the walk found, each name once, in order of appearance."""

    foreign: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    # The properties read off the filter variable itself (`resource.<name>`, `resource["<name>"]`),
    # so they can be checked against the resource type's own properties; a deeper access
    # (`resource.a.b`) contributes only its first name, the one the data schema can speak for.
    attributes: list[str] = field(default_factory=list)
    # `resource[<expression>]`, where the property read is only known once the expression is
    # evaluated - so it can't be matched against the schema's property list at all.
    computed_index: bool = False

    def note(self, names: list[str], name: str) -> None:
        if name not in names:
            names.append(name)


def _bare_identifier(argument: object) -> str | None:
    """The name if `argument` is exactly one identifier (a chain of single-child wrapper rules
    ending in `ident`), the only form a comprehension's loop variable can take - else None.
    """
    node = argument
    while isinstance(node, Tree) and node.data != "ident" and len(node.children) == 1:
        node = node.children[0]
    if isinstance(node, Tree) and node.data == "ident":
        return str(node.children[0])
    return None


def _walk_method_call(node: Tree, bound: frozenset[str], findings: _Findings) -> None:
    """Visit a receiver-style call `x.method(args)`: a blocked method is recorded, and a
    comprehension macro binds a bare-identifier first argument for its remaining arguments only.
    """
    # children: receiver, method name, and the argument list - absent for `x.size()`.
    receiver, method, *rest = node.children
    method_name = str(method)
    if method_name in _BLOCKED_FUNCTIONS:
        findings.note(findings.blocked, method_name)
    _walk(receiver, bound, findings)
    arguments = list(rest[0].children) if rest and isinstance(rest[0], Tree) else []
    loop_variable = _bare_identifier(arguments[0]) if method_name in _COMPREHENSION_MACROS and arguments else None
    if loop_variable is not None:
        bound = bound | {loop_variable}
        arguments = arguments[1:]
    for argument in arguments:
        _walk(argument, bound, findings)


def _string_literal(node: object) -> str | None:
    """The value of a string-literal node (`'email'` -> `email`), or None for anything else - a
    computed key, a number, a raw or triple-quoted literal this doesn't try to decode.
    """
    while isinstance(node, Tree) and node.data != "literal" and len(node.children) == 1:
        node = node.children[0]
    if not (isinstance(node, Tree) and node.data == "literal"):
        return None
    token = str(node.children[0])
    quoted = len(token) >= _QUOTE_PAIR and token[0] == token[-1] and token[0] in "'\""
    return token[1:-1] if quoted else None


def _note_member_access(node: Tree, findings: _Findings) -> None:
    """Record the property a `resource.x` / `resource["x"]` node reads. Anything else is skipped:
    a nested access (`resource.a.b`, whose receiver is itself a member access) contributes only
    its first name, and an access on another variable contributes nothing. An index whose key is
    an expression rather than a literal names a property only at evaluation time, so it is noted
    as computed instead - filter_errors reports it when it has a property list to check against.
    """
    receiver, key = node.children[0], node.children[1]
    if _bare_identifier(receiver) != FILTER_VARIABLE:
        return
    if node.data == _MEMBER_DOT_NODE:
        findings.note(findings.attributes, str(key))
        return
    name = _string_literal(key)
    if name is None:
        findings.computed_index = True
    else:
        findings.note(findings.attributes, name)


def _note_variable(node: Tree, bound: frozenset[str], findings: _Findings) -> None:
    """Record a variable reference that is neither the filter variable nor a loop variable."""
    name = str(node.children[0])
    if name != FILTER_VARIABLE and name not in bound:
        findings.note(findings.foreign, name)


def _walk_global_call(node: Tree, bound: frozenset[str], findings: _Findings) -> None:
    """Visit a global call `f(args)`, recording a blocked or non-CEL function by name."""
    function = str(node.children[0])
    if function in _BLOCKED_FUNCTIONS:
        findings.note(findings.blocked, function)
    elif function not in _GLOBAL_FUNCTIONS:
        findings.note(findings.unknown, function)
    for child in node.children[1:]:
        _walk(child, bound, findings)


def _walk(node: object, bound: frozenset[str], findings: _Findings) -> None:
    """Visit `node` with the loop variables in scope, recording foreign variables, blocked calls
    and the properties read off the filter variable.
    """
    if not isinstance(node, Tree):
        return
    if node.data in _VARIABLE_NODES:
        _note_variable(node, bound, findings)
    elif node.data in _GLOBAL_CALL_NODES:
        _walk_global_call(node, bound, findings)
    elif node.data == _METHOD_CALL_NODE:
        _walk_method_call(node, bound, findings)
    else:
        if node.data in _MEMBER_ACCESS_NODES:
            _note_member_access(node, findings)
        for child in node.children:
            _walk(child, bound, findings)


def _conjunction_terms(node: Tree) -> list[Tree]:
    """The operands of a top-level `&&` chain (the grammar nests it to the left); `[node]` otherwise."""
    while node.data in {"expr", "conditionalor"} and len(node.children) == 1:
        node = node.children[0]
    if node.data != "conditionaland":
        return [node]
    terms: list[Tree] = []
    for child in node.children:
        terms += _conjunction_terms(child) if isinstance(child, Tree) and child.data == "conditionaland" else [child]
    return terms


def conjuncts(expression: str) -> list[str] | None:
    """The top-level `&&` terms of `expression`, each exactly as written; None if it doesn't parse.
    Parsed, not split on the text: a call, a group or a string literal stays inside its term.
    """
    try:
        tree = _environment.compile(expression)
    except celpy.CELParseError:
        return None
    return [expression[term.meta.start_pos : term.meta.end_pos] for term in _conjunction_terms(tree)]


def filter_errors(expression: str, properties: set[str] | None = None) -> list[str]:
    """Every reason `expression` isn't an acceptable `context.filters` value; empty if it is.

    Messages are ready to be `reason` lines of errors.validation_feedback_error, in the same
    style as subject_binding's shape checks (they file under its "Malformed shape" heading).

    With `properties` - the resource type's own properties from the data schema - a filter that
    reads a property the type doesn't have is an error too: seen live, flash-lite invents one to
    widen a match (`resource.email == 'Volvo'` on a `Car`), which reads as a real condition
    downstream and can't be answered. An empty or absent set disables that check rather than
    failing every filter: a type the schema observed no properties for, or a schema that couldn't
    be fetched, is no evidence that a property doesn't exist.
    """
    if len(expression) > MAX_FILTER_LENGTH:
        return [f"context.filters is longer than {MAX_FILTER_LENGTH} characters"]
    try:
        tree = _environment.compile(expression)
    except celpy.CELParseError as exc:
        position = f" (line {exc.line}, column {exc.column})" if exc.line is not None else ""
        return [f"context.filters is not valid CEL{position}"]
    findings = _Findings()
    _walk(tree, frozenset(), findings)
    errors = [
        f"context.filters references {name!r}; only {FILTER_VARIABLE}.<property> may be referenced"
        for name in findings.foreign
    ]
    errors.extend(f"context.filters calls {function}(), which is not allowed" for function in findings.blocked)
    errors.extend(
        f"context.filters calls {function}(), not a CEL function (CEL has double(), int(), string(), ...)"
        for function in findings.unknown
    )
    if properties:
        known = properties | _ALWAYS_KNOWN_PROPERTIES
        errors.extend(
            f"context.filters reads {FILTER_VARIABLE}.{name}, which is not a property of the resource type"
            for name in findings.attributes
            if name not in known
        )
        if findings.computed_index:
            errors.append(
                f"context.filters indexes {FILTER_VARIABLE} with an expression instead of a property "
                "name, so what it reads can't be checked against the resource type"
            )
    return errors
