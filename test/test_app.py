"""Tests for app.py's build_app(): GetTask/ListTasks/CancelTask are rejected as unsupported -
IAG only ever calls SendMessage, so a task never needs to be looked up or canceled afterward.
"""

import jwt
from starlette.testclient import TestClient

from indykite.intent_agent.app import build_app
from indykite.intent_agent.config import Settings

_APP_AGENT_KEY = jwt.encode({"sub": "test-app-agent"}, key="", algorithm="none")


def _set_baseline_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("IK_APP_AGENT_KEY", _APP_AGENT_KEY)
    monkeypatch.setenv("INDYKITE_BASE_URL", "https://ikg.example.com")
    monkeypatch.setenv("AUDIT_DELIVERY", "file")
    monkeypatch.setenv("AUDIT_FILE_STORAGE_PATH", "./.private/audit-never-created")
    monkeypatch.setenv("AUDIT_FILE_FORMAT", "json")
    monkeypatch.setenv("AUDIT_FILE_ROTATION_STRATEGY", "time")


def test_get_task_list_tasks_cancel_task_are_rejected_as_unsupported(monkeypatch, mocker):
    """IAG only uses SendMessage, but GetTask/ListTasks/CancelTask are still A2A-exposed, so
    they must come back as a clean UNSUPPORTED_OPERATION JSON-RPC error, not reach the store.
    """
    _set_baseline_env(monkeypatch)
    fake_schema_client = mocker.Mock(preload=mocker.Mock(), aclose=mocker.AsyncMock())
    fake_policy_client = mocker.Mock(aclose=mocker.AsyncMock())
    settings = Settings(_env_file=None)
    app, _agent = build_app(settings, schema_client=fake_schema_client, policy_client=fake_policy_client)
    client = TestClient(app)

    for method, params in [
        ("GetTask", {"id": "whatever"}),
        ("ListTasks", {}),
        ("CancelTask", {"id": "whatever"}),
    ]:
        response = client.post(
            "/", headers={"A2A-Version": "1.0"}, json={"jsonrpc": "2.0", "id": "1", "method": method, "params": params}
        )
        error = response.json()["error"]
        assert error["message"] == f"{method} is not supported."
        assert error["data"][0]["reason"] == "UNSUPPORTED_OPERATION"
