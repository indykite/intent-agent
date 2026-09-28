"""Shared pytest fixtures: a fully populated audit record factory, a baseline Settings env, and
factories for golden eval fixture files. (Fixtures live here rather than in a helper module
because pre-commit's name-tests-test hook requires every other module under test/ to be a
test_*.py file.)
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from evals.fixtures import Fixture, load_fixture
from indykite.intent_agent.audit.message import AuditMessage, Outcome

_DEFAULTS: dict[str, Any] = {
    "timestamp": "2026-09-07T10:15:04.123456Z",
    "request_id": "req-1",
    "subject_id": "user-1",
    "subject_type": "Customer",
    "requesting_agent_id": "",
    "context_id": "ctx-1",
    "prompt": "rent kitt",
    "evaluations": [{"action": {"name": "RENT"}, "resource": {"type": "Car", "id": "kitt"}}],
    "schema_version": "abc123",
    "policy_version": "",
    "llm_provider": "gemini",
    "llm_model": "gemini-3.5-flash-lite",
    "latency_ms": 42,
    "validation_retries": 0,
    "policy_fetch_failed": False,
    "outcome": Outcome.SUCCESS,
    "message": "",
}


@pytest.fixture
def audit_message():
    """A factory for a complete audit record, overriding whatever the test cares about."""

    def _build(**overrides: Any) -> AuditMessage:
        return AuditMessage(**{**_DEFAULTS, **overrides})

    return _build


@pytest.fixture
def baseline_env(monkeypatch):
    """A function setting every required Settings env var, with file audit delivery into the
    given directory (a never-created placeholder by default - Settings never touches it).
    """

    def _set(audit_storage_path="./.private/audit-never-created") -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
        monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
        monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
        monkeypatch.setenv("IK_APP_AGENT_KEY", "test-app-agent-key")
        monkeypatch.setenv("INDYKITE_BASE_URL", "https://ikg.example.com")
        monkeypatch.setenv("AUDIT_DELIVERY", "file")
        monkeypatch.setenv("AUDIT_FILE_STORAGE_PATH", str(audit_storage_path))
        monkeypatch.setenv("AUDIT_FILE_FORMAT", "json")
        monkeypatch.setenv("AUDIT_FILE_ROTATION_STRATEGY", "time")

    return _set


def _jgf_node(*, digital_twin: bool) -> dict[str, Any]:
    """A JGF node entry the way GET /data-schema/v1 shapes it, with or without the DigitalTwin label."""
    labels = [{"name": "DigitalTwin", "count": 1}] if digital_twin else [{"name": "Resource", "count": 1}]
    return {"metadata": {"properties": {}, "system_labels": labels, "node_count": 1}}


_EVAL_FIXTURE_DEFAULTS: dict[str, Any] = {
    "prompt": "Can I access the document doc123?",
    "jwt_sub": "leslie",
    "category": "single intent",
    "data_model": "user_model",
}
_EVAL_EVALUATION = {
    "subject": {"type": "User", "id": "leslie"},
    "action": {"name": "CAN_ACCESS"},
    "resource": {"type": "Document", "id": "doc123"},
    "context": {},
}


@pytest.fixture
def eval_schema_file(tmp_path):
    """A factory writing a small JGF data model under `<directory>/schema/<model>_schema.json`
    (directory defaults to tmp_path/evals) and returning its path.
    """

    def _write(
        model: str = "user_model",
        *,
        digital_twins: Iterable[str] = ("User",),
        resources: Iterable[str] = ("Document",),
        relations: Iterable[str | tuple[str, str, str]] = ("CAN_ACCESS",),
        directory: Path | None = None,
    ) -> Path:
        """A relation given as a bare name is a User -> Document edge; a (source, relation, target)
        triple is that edge.
        """
        nodes = {"_Application": _jgf_node(digital_twin=True)}
        nodes.update({label: _jgf_node(digital_twin=True) for label in digital_twins})
        nodes.update({label: _jgf_node(digital_twin=False) for label in resources})
        triples = [("User", relation, "Document") if isinstance(relation, str) else relation for relation in relations]
        edges = [
            {"source": source, "relation": relation, "target": target, "directed": True}
            for source, relation, target in triples
        ]
        schema_dir = (directory or tmp_path / "evals") / "schema"
        schema_dir.mkdir(parents=True, exist_ok=True)
        path = schema_dir / f"{model}_schema.json"
        path.write_text(json.dumps({"graph": {"nodes": nodes, "edges": edges}}), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def eval_policies_file(tmp_path):
    """A factory writing `<directory>/schema/<model>_policies.json` (directory defaults to
    tmp_path/evals) from (subject type, action, resource type) triples, in the shape of
    `GET /access/v1/policies`, and returning its path.
    """

    def _write(model: str, triples: Iterable[tuple[str, str, str]], *, directory: Path | None = None) -> Path:
        results = [
            {"policy": {"subject": {"type": s}, "actions": [a], "resource": {"type": r}}, "tags": []}
            for s, a, r in triples
        ]
        schema_dir = (directory or tmp_path / "evals") / "schema"
        schema_dir.mkdir(parents=True, exist_ok=True)
        path = schema_dir / f"{model}_policies.json"
        path.write_text(json.dumps({"results": results}), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def eval_fixture_file(tmp_path, eval_schema_file):
    """A factory writing one golden fixture file `<directory>/<name>.json` (directory defaults to
    tmp_path/evals, with a `user_model` schema alongside unless one already exists) and returning
    its path. `fields` override the fixture document's keys; `expected` defaults to one matching
    evaluation for the default prompt.
    """

    def _write(name: str = "single_intent_test_1", *, directory: Path | None = None, **fields: Any) -> Path:
        directory = directory or tmp_path / "evals"
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "schema" / "user_model_schema.json").exists():
            eval_schema_file(directory=directory)
        document = {**_EVAL_FIXTURE_DEFAULTS, **fields}
        document.setdefault("expected", {"prompt": document["prompt"], "evaluations": [_EVAL_EVALUATION]})
        path = directory / f"{name}.json"
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def eval_fixture(eval_fixture_file):
    """A factory for a loaded, validated golden Fixture (see eval_fixture_file for the defaults)."""

    def _build(name: str = "single_intent_test_1", **fields: Any) -> Fixture:
        return load_fixture(eval_fixture_file(name, **fields))

    return _build
