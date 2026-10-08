"""Golden eval fixtures (TDD 7.2): one JSON file per prompt under this directory, loaded and
validated up front so a malformed fixture fails the run before any LLM call is spent.

A fixture is `{prompt, jwt_sub, expected, category, data_model[, subject_type]}`: `expected` is
the complete response the agent must return for `prompt` - either `{prompt, evaluations}` or the
4.7 error shape `{prompt, error, reason, obligation}` for a prompt that must be refused;
`data_model` names a schema under `schema/` (`<data_model>_schema.json`) served to the agent in
place of the platform's `GET /data-schema/v1`, plus, when present, the access policies of that
model (`<data_model>_policies.json`, in the shape of `GET /access/v1/policies`) served in place
of the platform's policy listing; `jwt_sub` is the caller's id the agent binds, as IAG supplies
it per request; `subject_type` is the caller's node type - optional when the data model has
exactly one DigitalTwin type (it is that type), required when it has several; `confidence`
(optional) is the rating the fixture expects the model to give, taken as written and only
reported. Every action a golden answer names must be one a policy of the caller's subject type
grants on that evaluation's resource type - graph edges are never action names, the same rule
the pipeline applies at run time - so a fixture naming anything else is rejected at load time
instead of reading as a model failure.
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

# TDD 7.2's fixture categories, in report order. Fixtures spell them freely
# ("single-intent", "Subject Search", "adversarial_subject_spoof"); canonical_category maps every
# spelling onto one of these, and any other value is a load error - a mis-spelt category would
# otherwise silently drop the fixture out of the gate that depends on it (see scoring).
SINGLE_INTENT = "single intent"
MULTI_INTENT = "multi intent"
SUBJECT_SEARCH = "subject search"
RESOURCE_SEARCH = "resource search"
AMBIGUOUS = "ambiguous"
OUT_OF_SCOPE = "out of scope"
# The adversarial categories (TDD 7.2: subject spoof, injection, oversized resource.id) all feed
# the injection-resistance gate - see is_adversarial.
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

# The platform's policy listing shape (GET /access/v1/policies): {"results": [{"policy": {...},
# "tags": [...]}, ...]}. The pipeline works on the `results` entries.
NO_POLICIES: list[dict[str, Any]] = []


def is_adversarial(category: str) -> bool:
    """True for every adversarial category - the fixtures the injection-resistance gate covers."""
    return category.startswith(_ADVERSARIAL_PREFIX)


# Every adversarial category, in report order - the injection-resistance gate requires each one
# to be present in a run, so a category can't drop out of the suite unnoticed.
ADVERSARIAL_CATEGORIES = tuple(category for category in CATEGORIES if is_adversarial(category))
# The 4.7 error shape (errors.error_response) - a golden refusal and a judged refusal alike.
REFUSAL_KEYS = frozenset({"prompt", "error", "reason", "obligation"})


def policy_entries(document: object) -> list[dict[str, Any]]:
    """The `results` entries of a policy listing document; `[]` for anything else (load_policies
    has already rejected a file that isn't shaped as a listing).
    """
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        return NO_POLICIES
    return [entry for entry in document["results"] if isinstance(entry, dict)]


def policies_for_subject(entries: list[dict[str, Any]], subject_type: str) -> list[dict[str, Any]]:
    """The listing entries whose policy subject type is `subject_type` - the narrowing the
    platform's `GET /access/v1/policies?subject_type=` applies, so what the runner serves for a
    fixture's caller (runner.FixturePolicyClient) and what the loader checks its golden answer
    against (_check_actions_producible) is one and the same set.
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
    # The rating the fixture expects, as written - only shown in the report.
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
    """Map a fixture's category spelling onto CATEGORIES: case, `-`/`_` and repeated spaces
    don't matter. Raises ValueError for anything that isn't one of them.
    """
    canonical = re.sub(r"[\s_-]+", " ", value).strip().lower()
    if canonical not in CATEGORIES:
        message = f"unknown category {value!r}; expected one of: {', '.join(CATEGORIES)}"
        raise ValueError(message)
    return canonical


def _load_json_object(path: Path, what: str) -> dict[str, Any]:
    """Parse `path` as a JSON object, or raise FixtureError naming the file - the same error path
    as a malformed fixture, so a bad data-model or policy file is a usage error (exit 2), not a
    traceback.
    """
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
    """The JGF data schema at `path`, parsed once per process (fixtures share a handful of models).
    It must be an object with a `graph` object holding `nodes` (object) and `edges` (array) - the
    shape every reader in the pipeline expects.
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
    """The policy listing at `path` (the `GET /access/v1/policies` shape: an object whose
    `results` is an array of `{"policy": {...}, "tags": [...]}` entries), parsed once per process.
    """
    document = _load_json_object(path, "policy listing")
    results = document.get("results")
    if not isinstance(results, list) or not all(
        isinstance(entry, dict) and isinstance(entry.get("policy"), dict) for entry in results
    ):
        message = f"{path.name}: policy listing must have a 'results' array of {{\"policy\": ...}} objects"
        raise FixtureError(message)
    return document


def _as_model_output(fixture: Fixture) -> list[dict[str, Any]]:
    """The golden evaluations as the model would have to produce them for the binding to yield
    them: a subject that isn't the caller carries the `subject_source` marker back - "search"
    for a bare type (a Subject Search), "prompt" for an identified second subject - since the
    response strips the markers but the checks below key on them. Copies; the fixture's own
    evaluations are never touched. An anonymous caller (blank `jwt_sub`) is bound without an id,
    so its own evaluations read as searches here - the lenient reading, never a false rejection.
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
    """Reject a golden answer the pipeline could never return: first every action against the
    caller's policies (for a precise message), then a round trip through normalize_intent_response.
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
    """Feed the golden evaluations, markers restored, through the real normalisation as the
    fixture's caller and require the response to equal `expected`. A rejection means the
    pipeline would feed this answer back to the model; a different response means the pipeline
    rewrites it (rebinds a subject the gate doesn't cover, completes a marker, drops a search
    over a non-DigitalTwin type or a duplicate, strips a blank id ...) - either way the fixture
    could only ever read as a model failure.
    """
    model_output = json.dumps(
        # The rating never changes the response, so any valid level stands in for the model's.
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

    It is interpolated into the schema and policy file names, so anything with a path separator or
    a `..` segment would name a file outside `schema/`: `../outside` resolves and loads today. The
    files it can reach are narrow (a `*_schema.json` that parses as JGF, a `*_policies.json` in the
    listing shape) and a fixture author can already edit the runner itself, so this is input
    validation rather than a boundary - the same reason `canonical_category` refuses an unknown
    category: a fixture must not quietly read something other than the model it names.
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
    """Why this golden evaluation isn't a shape the pipeline can return, or None. Mirrors what
    subject_binding rejects or drops: `resource` must be an object with a non-blank string
    `type`; `action`, when present (absent is the Action Search shape), an object with a
    non-blank string `name` (a present-but-blank name is dropped, never returned); `subject`,
    when present, an object; `context`, when present, an object whose `parameters` and
    `qualifiers` are objects and whose `filters` is a string.
    """
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
    """The caller's node type: the fixture's own `subject_type`, else the data model's single
    DigitalTwin type. Either way it must be a node type of the data model, or run_extraction
    would refuse the request before the LLM ever saw the prompt.
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
    """Parse and validate one fixture file - every problem is a FixtureError naming the file.
    Data models are looked up under `schema_dir`, by default the `schema/` directory next to
    the fixture.
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
    """Every `*.json` fixture directly under `directory` (schemas live in `schema/`, not here),
    sorted by id. An empty directory is a FixtureError: a suite with nothing in it can't gate
    anything.
    """
    paths = sorted(path for path in directory.glob("*.json") if path.is_file())
    if not paths:
        message = f"no fixture files (*.json) found under {directory}"
        raise FixtureError(message)
    schema_dir = schema_dir if schema_dir is not None else directory / "schema"
    return [load_fixture(path, schema_dir=schema_dir) for path in paths]
