"""Shared pytest fixtures for the audit tests: a fully populated record factory and a baseline
Settings env. (Fixtures live here rather than in a helper module because pre-commit's
name-tests-test hook requires every other module under test/ to be a test_*.py file.)
"""

from typing import Any

import pytest

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
