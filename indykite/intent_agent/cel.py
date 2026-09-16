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
# Comprehension macros bind their first argument as a loop variable, a legitimate identifier
# inside the macro's remaining arguments only (`resource.tags.exists(t, t == 'x')`).
_COMPREHENSION_MACROS = frozenset({"all", "exists", "exists_one", "filter", "map"})
# Generous for a prompt-derived condition; a filter past it is runaway output, not intent.
MAX_FILTER_LENGTH = 1000

# celpy's grammar rule names: a variable reference, with or without CEL's leading-dot absolute
# form (`.subject.id`), and a global function call in the same two spellings.
_VARIABLE_NODES = frozenset({"ident", "dot_ident"})
_GLOBAL_CALL_NODES = frozenset({"ident_arg", "dot_ident_arg"})
_METHOD_CALL_NODE = "member_dot_arg"

_environment = celpy.Environment()


@dataclass
class _Findings:
    """What the walk found, each name once, in order of appearance."""

    foreign: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)

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


def _walk(node: object, bound: frozenset[str], findings: _Findings) -> None:
    """Visit `node` with the loop variables in scope, recording foreign variables and blocked calls."""
    if not isinstance(node, Tree):
        return
    if node.data in _VARIABLE_NODES:
        name = str(node.children[0])
        if name != FILTER_VARIABLE and name not in bound:
            findings.note(findings.foreign, name)
    elif node.data in _GLOBAL_CALL_NODES:
        function = str(node.children[0])
        if function in _BLOCKED_FUNCTIONS:
            findings.note(findings.blocked, function)
        for child in node.children[1:]:
            _walk(child, bound, findings)
    elif node.data == _METHOD_CALL_NODE:
        _walk_method_call(node, bound, findings)
    else:
        for child in node.children:
            _walk(child, bound, findings)


def filter_errors(expression: str) -> list[str]:
    """Every reason `expression` isn't an acceptable `context.filters` value; empty if it is.

    Messages are ready to be `reason` lines of errors.validation_feedback_error, in the same
    style as subject_binding's shape checks (they file under its "Malformed shape" heading).
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
    return errors
