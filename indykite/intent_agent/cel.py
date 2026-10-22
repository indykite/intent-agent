"""CEL checks for `context.filters`: parse, one allowed variable, no reflection, size cap.

Filters are never evaluated here (the IBAC Engine does that); failures feed the LLM retry loop.
"""

from dataclasses import dataclass, field

import celpy
from lark import Tree

FILTER_VARIABLE = "resource"
# Like the platform's evaluator: no host-type reflection, as `dyn(x)` or `x.dyn()`.
_BLOCKED_FUNCTIONS = frozenset({"dyn", "type"})
# The parser accepts any name, so an invented `float(x)` would otherwise only fail inside IBAC.
_GLOBAL_FUNCTIONS = frozenset(
    {"bool", "bytes", "double", "duration", "has", "int", "matches", "size", "string", "timestamp", "uint"}
)
# These bind their first argument as a loop variable (`resource.tags.exists(t, t == 'x')`).
_COMPREHENSION_MACROS = frozenset({"all", "exists", "exists_one", "filter", "map"})
# Generous for a prompt-derived condition; a filter past it is runaway output, not intent.
MAX_FILTER_LENGTH = 1000
_QUOTE_PAIR = 2

# celpy grammar rule names, each with and without CEL's leading-dot absolute form.
_VARIABLE_NODES = frozenset({"ident", "dot_ident"})
_GLOBAL_CALL_NODES = frozenset({"ident_arg", "dot_ident_arg"})
_METHOD_CALL_NODE = "member_dot_arg"
# `resource.email` and `resource["email"]` mean the same, so the property check must see both.
_MEMBER_DOT_NODE = "member_dot"
_MEMBER_INDEX_NODE = "member_index"
_MEMBER_ACCESS_NODES = frozenset({_MEMBER_DOT_NODE, _MEMBER_INDEX_NODE})

# AuthZEN resource fields, never in the data schema's property list.
_ALWAYS_KNOWN_PROPERTIES = frozenset({"id", "type"})

_environment = celpy.Environment()


@dataclass
class _Findings:
    """What the walk found, each name once, in order of appearance."""

    foreign: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    # Properties read off the filter variable; `resource.a.b` contributes only `a`.
    attributes: list[str] = field(default_factory=list)
    # `resource[<expression>]`: the property read can't be matched against the schema.
    computed_index: bool = False

    def note(self, names: list[str], name: str) -> None:
        if name not in names:
            names.append(name)


def _bare_identifier(argument: object) -> str | None:
    """Return the name if `argument` is exactly one identifier, else None."""
    node = argument
    while isinstance(node, Tree) and node.data != "ident" and len(node.children) == 1:
        node = node.children[0]
    if isinstance(node, Tree) and node.data == "ident":
        return str(node.children[0])
    return None


def _walk_method_call(node: Tree, bound: frozenset[str], findings: _Findings) -> None:
    """Visit `x.method(args)`, recording blocked methods and binding comprehension loop variables."""
    # The argument list is absent for `x.size()`.
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
    """Return a plain quoted string literal's value (`'email'` -> `email`), else None."""
    while isinstance(node, Tree) and node.data != "literal" and len(node.children) == 1:
        node = node.children[0]
    if not (isinstance(node, Tree) and node.data == "literal"):
        return None
    token = str(node.children[0])
    quoted = len(token) >= _QUOTE_PAIR and token[0] == token[-1] and token[0] in "'\""
    return token[1:-1] if quoted else None


def _note_member_access(node: Tree, findings: _Findings) -> None:
    """Record the property a `resource.x` / `resource["x"]` node reads, or flag a computed index."""
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
    """Visit `node` with `bound` loop variables in scope, collecting into `findings`."""
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
    """Return every reason `expression` isn't an acceptable `context.filters` value; empty if it is.

    With `properties` (the resource type's schema properties), reading any other property is an
    error too; an empty or absent set disables that check, as it is no evidence of absence.
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
