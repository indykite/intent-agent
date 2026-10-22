"""Golden eval fixtures: loaded and validated up front, so a malformed one fails the run before any LLM call.

A fixture is `{prompt, jwt_sub, expected, category, data_model[, subject_type, confidence]}` (see
docs/testing.md); a golden answer the pipeline could never produce is rejected at load time.
"""

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from indykite.intent_agent.errors import IntentValidationError
from indykite.intent_agent.input_schema import CONFIDENCE_LEVELS, _node_types, digital_twin_types, policy_triples
from indykite.intent_agent.subject_binding import (
    PolicyContext,
    RequestSubject,
    normalize_intent_response,
)

FIXTURES_DIR = Path(__file__).parent
SCHEMA_DIR = FIXTURES_DIR / "schema"

# In report order. An unknown spelling is a load error, so a fixture can't silently drop out of a gate.
SINGLE_INTENT = "single intent"
MULTI_INTENT = "multi intent"
SUBJECT_SEARCH = "subject search"
RESOURCE_SEARCH = "resource search"
AMBIGUOUS = "ambiguous"
OUT_OF_SCOPE = "out of scope"
# Every adversarial category feeds the injection-resistance gate.
ADVERSARIAL_SUBJECT_SPOOF = "adversarial subject spoof"
ADVERSARIAL_SQL_INJECTION = "adversarial sql injection"
ADVERSARIAL_OVERSIZED_INPUT = "adversarial oversized input"
CATEGORIES = (
    SINGLE_INTENT,
    MULTI_INTENT,
    SUBJECT_SEARCH,
    RESOURCE_SEARCH,
    AMBIGUOUS,
    OUT_OF_SCOPE,
    ADVERSARIAL_SUBJECT_SPOOF,
    ADVERSARIAL_SQL_INJECTION,
    ADVERSARIAL_OVERSIZED_INPUT,
)
_ADVERSARIAL_PREFIX = "adversarial "
__all__ = [
    "FIXTURES_DIR",
    "Fixture",
    "FixtureError",
    "digital_twin_types",
    "load_fixture",
    "load_fixtures",
    "load_schema",
]

NO_POLICIES: list[dict[str, Any]] = []


def is_adversarial(category: str) -> bool:
    """True for every adversarial category - the fixtures the injection-resistance gate covers."""
    return category.startswith(_ADVERSARIAL_PREFIX)


# A full run must contain each of these, so a category can't drop out of the suite unnoticed.
ADVERSARIAL_CATEGORIES = tuple(category for category in CATEGORIES if is_adversarial(category))
# The error shape (errors.error_response) - a golden refusal and a judged refusal alike.
REFUSAL_KEYS = frozenset({"prompt", "error", "reason", "obligation"})


def policy_entries(document: object) -> list[dict[str, Any]]:
    """The `results` entries of a policy listing document; `[]` for anything else."""
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        return NO_POLICIES
    return [entry for entry in document["results"] if isinstance(entry, dict)]


def policies_for_subject(entries: list[dict[str, Any]], subject_type: str) -> list[dict[str, Any]]:
    """The entries whose policy subject type is `subject_type`, as `?subject_type=` narrows the listing.

    Shared by the runner and the loader, so both see the same set.
    """
    return [
        entry
        for entry in entries
        if isinstance(entry.get("policy"), dict)
        and isinstance(entry["policy"].get("subject"), dict)
        and entry["policy"]["subject"].get("type") == subject_type
    ]


_DIGITAL_TWIN_LABEL = "DigitalTwin"


class FixtureError(ValueError):
    """A fixture file that can't be used as written - the message names the file and the problem."""


@dataclass(frozen=True, slots=True)
class Fixture:
    """One golden prompt with the response the agent must produce for it."""

    id: str
    path: Path
    prompt: str
    jwt_sub: str
    subject_type: str
    category: str
    data_model: str
    schema_path: Path
    expected: dict[str, Any]
    policies_path: Path | None = None
    # Only shown in the report, never judged.
    confidence: str | None = None

    @property
    def expects_refusal(self) -> bool:
        """True when the golden answer is the no-intent error rather than evaluations."""
        return "evaluations" not in self.expected

    @property
    def policies(self) -> list[dict[str, Any]]:
        """The data model's policy entries (`results` of the listing), `[]` without a policies file."""
        return policy_entries(load_policies(self.policies_path)) if self.policies_path is not None else NO_POLICIES


def canonical_category(value: str) -> str:
    """Map a category spelling onto CATEGORIES, ignoring case, `-`/`_` and repeated spaces.

    Raises ValueError for anything else.
    """
    canonical = re.sub(r"[\s_-]+", " ", value).strip().lower()
    if canonical not in CATEGORIES:
        message = f"unknown category {value!r}; expected one of: {', '.join(CATEGORIES)}"
        raise ValueError(message)
    return canonical


def _load_json_object(path: Path, what: str) -> dict[str, Any]:
    """Parse `path` as a JSON object; raises FixtureError naming the file (a usage error, not a traceback)."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        message = f"{path.name}: {what} can't be read ({exc.strerror or exc})"
        raise FixtureError(message) from exc
    except json.JSONDecodeError as exc:
        message = f"{path.name}: {what} is not valid JSON ({exc.msg} at line {exc.lineno})"
        raise FixtureError(message) from exc
    if not isinstance(document, dict):
        message = f"{path.name}: {what} must be a JSON object"
        raise FixtureError(message)
    return document


@cache
def load_schema(path: Path) -> dict[str, Any]:
    """The JGF data schema at `path`, parsed once per process.

    Raises FixtureError unless it has a `graph` object with `nodes` (object) and `edges` (array).
    """
    document = _load_json_object(path, "data model schema")
    graph = document.get("graph")
    if (
        not isinstance(graph, dict)
        or not isinstance(graph.get("nodes"), dict)
        or not isinstance(graph.get("edges"), list)
    ):
        message = f"{path.name}: data model schema must have a 'graph' object with 'nodes' (object) and 'edges' (array)"
        raise FixtureError(message)
    return document


@cache
def load_policies(path: Path) -> dict[str, Any]:
    """The policy listing at `path` (`{"results": [{"policy": ..., "tags": ...}]}`), parsed once per process."""
    document = _load_json_object(path, "policy listing")
    results = document.get("results")
    if not isinstance(results, list) or not all(
        isinstance(entry, dict) and isinstance(entry.get("policy"), dict) for entry in results
    ):
        message = f"{path.name}: policy listing must have a 'results' array of {{\"policy\": ...}} objects"
        raise FixtureError(message)
    return document


def _as_model_output(fixture: Fixture) -> list[dict[str, Any]]:
    """Copies of the golden evaluations as the model must emit them, `subject_source` markers restored.

    A non-caller subject gets "prompt" when it has an id, else "search".
    """
    caller = (fixture.subject_type, fixture.jwt_sub)
    reconstructed: list[dict[str, Any]] = []
    for evaluation in fixture.expected.get("evaluations", []):
        subject = evaluation.get("subject")
        copy = json.loads(json.dumps(evaluation))
        if isinstance(subject, dict) and (subject.get("type"), subject.get("id")) != caller:
            source = "prompt" if subject.get("id") else "search"
            copy.setdefault("context", {}).setdefault("parameters", {})["subject_source"] = source
        reconstructed.append(copy)
    return reconstructed


def _check_actions_producible(fixture: Fixture) -> None:
    """Raise FixtureError for a golden answer the pipeline could never return.

    Actions are checked against the caller's policies first, for a precise message, then a round trip.
    """
    schema = load_schema(fixture.schema_path)
    policies = policies_for_subject(fixture.policies, fixture.subject_type)
    granted = {(action, resource) for _, action, resource in policy_triples(policies, schema)}
    for index, evaluation in enumerate(fixture.expected.get("evaluations", [])):
        action, resource = evaluation.get("action"), evaluation.get("resource")
        name = action.get("name") if isinstance(action, dict) else None
        resource_type = resource.get("type") if isinstance(resource, dict) else None
        if not isinstance(name, str) or (name, resource_type) in granted:
            continue
        message = (
            f"{fixture.path.name}: evaluations[{index}] names action {name!r}, which no policy in "
            f"{fixture.data_model}_policies.json grants subject type {fixture.subject_type!r} on resource type "
            f"{resource_type!r} - the pipeline can't produce it"
        )
        raise FixtureError(message)
    if not fixture.expects_refusal:
        _check_round_trip(fixture, schema, policies)


def _check_round_trip(fixture: Fixture, schema: dict[str, Any], policies: list[dict[str, Any]]) -> None:
    """Raise FixtureError unless the real normalisation, as the fixture's caller, returns `expected`.

    A rejected or rewritten golden answer could only ever read as a model failure.
    """
    model_output = json.dumps(
        # The rating never changes the response, so any valid level will do.
        {"evaluations": _as_model_output(fixture), "confidence": CONFIDENCE_LEVELS[0], "explanation": "golden answer"}
    )
    caller = RequestSubject(id=fixture.jwt_sub or None, type=fixture.subject_type)
    try:
        response = json.loads(
            normalize_intent_response(
                model_output,
                fixture.prompt,
                schema=schema,
                subject=caller,
                policy_context=PolicyContext(policies=policies, fetch_failed=False),
            )
        )
    except IntentValidationError as exc:
        message = (
            f"{fixture.path.name}: the pipeline rejects this golden answer ({exc.errors[0]}) - it can't be the response"
        )
        raise FixtureError(message) from exc
    if response == fixture.expected:
        return
    expected_evaluations = fixture.expected["evaluations"]
    actual_evaluations = response.get("evaluations", [])
    differing = next(
        (index for index, (a, b) in enumerate(zip(expected_evaluations, actual_evaluations, strict=False)) if a != b),
        min(len(expected_evaluations), len(actual_evaluations)),
    )
    if "evaluations" not in response:
        would_be = f"the response would be a refusal ({response.get('reason')})"
    elif differing < len(actual_evaluations):
        would_be = (
            f"evaluations[{differing}] would come back as {json.dumps(actual_evaluations[differing], sort_keys=True)}"
        )
    else:
        would_be = f"evaluations[{differing}] would not come back at all"
    message = f"{fixture.path.name}: the pipeline returns a different response for this golden answer - {would_be}"
    raise FixtureError(message)


def _require_str(raw: dict[str, Any], key: str, path: Path, *, allow_blank: bool = False) -> str:
    """`raw[key]` as a string, or FixtureError when missing, not a string or (unless allowed) blank."""
    value = raw.get(key)
    if not isinstance(value, str) or (not allow_blank and not value.strip()):
        message = f"{path.name}: {key!r} must be a non-empty string"
        raise FixtureError(message)
    return value


# A bare model name: one path segment, so no separator and no `..` can reach outside schema/.
_DATA_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _require_data_model(raw: dict[str, Any], path: Path) -> str:
    """`raw["data_model"]` as a bare model name - the stem of a file directly under `schema/`.

    It is interpolated into file names, so a separator or `..` would read a file outside `schema/`.
    """
    data_model = _require_str(raw, "data_model", path)
    if not _DATA_MODEL.match(data_model):
        message = (
            f"{path.name}: data model {data_model!r} is not a bare model name - it names a file "
            f"directly under {path.parent.name}/schema/, so letters, digits, '_' and '-' only"
        )
        raise FixtureError(message)
    return data_model


def _evaluation_shape_problem(evaluation: dict[str, Any]) -> str | None:
    """Why this golden evaluation isn't a shape the pipeline can return (mirrors subject_binding), or None."""
    resource = evaluation.get("resource")
    if not isinstance(resource, dict) or not isinstance(resource.get("type"), str) or not resource["type"].strip():
        return "'resource' must be an object with a non-empty string 'type'"
    action = evaluation.get("action")
    if "action" in evaluation and (
        not isinstance(action, dict) or not isinstance(action.get("name"), str) or not action["name"].strip()
    ):
        return "'action', when present, must be an object with a non-empty string 'name'"
    if "subject" in evaluation and not isinstance(evaluation["subject"], dict):
        return "'subject', when present, must be an object"
    context = evaluation.get("context")
    if "context" in evaluation:
        if not isinstance(context, dict):
            return "'context', when present, must be an object"
        for key, kind in (("parameters", dict), ("qualifiers", dict), ("filters", str)):
            if key in context and not isinstance(context[key], kind):
                return f"'context.{key}', when present, must be {'an object' if kind is dict else 'a string'}"
    return None


def _validate_expected(expected: object, prompt: str, path: Path) -> dict[str, Any]:
    """Check `expected` is one of the two response shapes the agent can produce, echoing `prompt`."""
    if not isinstance(expected, dict):
        message = f"{path.name}: 'expected' must be an object"
        raise FixtureError(message)
    if "evaluations" in expected:
        evaluations = expected["evaluations"]
        if not isinstance(evaluations, list) or not evaluations or not all(isinstance(e, dict) for e in evaluations):
            message = f"{path.name}: 'expected.evaluations' must be a non-empty array of objects"
            raise FixtureError(message)
        for index, evaluation in enumerate(evaluations):
            problem = _evaluation_shape_problem(evaluation)
            if problem:
                message = f"{path.name}: 'expected.evaluations[{index}]': {problem}"
                raise FixtureError(message)
    elif set(expected) != REFUSAL_KEYS:
        message = (
            f"{path.name}: 'expected' must be either {{prompt, evaluations}} or the error shape "
            f"{{prompt, error, reason, obligation}}"
        )
        raise FixtureError(message)
    if expected.get("prompt") != prompt:
        message = f"{path.name}: 'expected.prompt' must echo 'prompt' verbatim"
        raise FixtureError(message)
    return expected


def _resolve_subject_type(raw: dict[str, Any], schema: dict[str, Any], path: Path) -> str:
    """The caller's node type: the fixture's `subject_type`, else the data model's single DigitalTwin type.

    Raises FixtureError unless it is a node type of the data model.
    """
    node_types = set(_node_types(schema))
    if "subject_type" in raw:
        subject_type = _require_str(raw, "subject_type", path)
    else:
        candidates = digital_twin_types(schema)
        if len(candidates) != 1:
            message = (
                f"{path.name}: 'subject_type' is required - data model {raw['data_model']!r} has "
                f"{len(candidates)} DigitalTwin types ({', '.join(candidates) or 'none'}), so the caller's "
                "type can't be inferred"
            )
            raise FixtureError(message)
        subject_type = candidates[0]
    if subject_type not in node_types:
        message = f"{path.name}: subject_type {subject_type!r} is not a node type of data model {raw['data_model']!r}"
        raise FixtureError(message)
    return subject_type


def load_fixture(path: Path, *, schema_dir: Path | None = None) -> Fixture:
    """Parse and validate one fixture file; every problem is a FixtureError naming the file.

    Data models are looked up under `schema_dir` (default: `schema/` next to the fixture).
    """
    schema_dir = schema_dir if schema_dir is not None else path.parent / "schema"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        message = f"{path.name}: not valid JSON ({exc.msg} at line {exc.lineno})"
        raise FixtureError(message) from exc
    if not isinstance(raw, dict):
        message = f"{path.name}: the fixture must be a JSON object"
        raise FixtureError(message)

    prompt = _require_str(raw, "prompt", path)
    jwt_sub = _require_str(raw, "jwt_sub", path, allow_blank=True)
    try:
        category = canonical_category(_require_str(raw, "category", path))
    except ValueError as exc:
        message = f"{path.name}: {exc}"
        raise FixtureError(message) from exc
    data_model = _require_data_model(raw, path)
    schema_path = schema_dir / f"{data_model}_schema.json"
    if not schema_path.is_file():
        message = f"{path.name}: data model {data_model!r} has no schema file at {schema_path}"
        raise FixtureError(message)
    schema = load_schema(schema_path)
    policies_path = schema_dir / f"{data_model}_policies.json"
    if policies_path.is_file():
        load_policies(policies_path)  # validated here, so a malformed listing fails the load, not the run
    expected = _validate_expected(raw.get("expected"), prompt, path)
    fixture = Fixture(
        id=path.stem,
        path=path,
        prompt=prompt,
        jwt_sub=jwt_sub,
        subject_type=_resolve_subject_type(raw, schema, path),
        category=category,
        data_model=data_model,
        schema_path=schema_path,
        expected=expected,
        policies_path=policies_path if policies_path.is_file() else None,
        confidence=raw.get("confidence"),
    )
    _check_actions_producible(fixture)
    return fixture


def load_fixtures(directory: Path = FIXTURES_DIR, *, schema_dir: Path | None = None) -> list[Fixture]:
    """Every `*.json` fixture directly under `directory`, sorted by id.

    Raises FixtureError for an empty directory: an empty suite can't gate anything.
    """
    paths = sorted(path for path in directory.glob("*.json") if path.is_file())
    if not paths:
        message = f"no fixture files (*.json) found under {directory}"
        raise FixtureError(message)
    schema_dir = schema_dir if schema_dir is not None else directory / "schema"
    return [load_fixture(path, schema_dir=schema_dir) for path in paths]
