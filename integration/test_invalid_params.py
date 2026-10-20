"""A request with invalid subject metadata comes back as a JSON-RPC error in the A2A envelope, not
as a plain-text HTTP response.
"""

import json
import logging

import pytest
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from indykite.intent_agent import auth
from indykite.intent_agent.agent import IntentAgent
from indykite.intent_agent.app import build_app
from indykite.intent_agent.audit.message import Outcome
from indykite.intent_agent.config import Settings

_INVALID_PARAMS = -32602


def _set_baseline_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("INDYKITE_BASE_URL", "https://ikg.example.com")
    monkeypatch.setenv("IK_APP_AGENT_KEY", "test-app-agent-key")
    monkeypatch.setenv("AUDIT_DELIVERY", "file")
    monkeypatch.setenv("AUDIT_FILE_STORAGE_PATH", str(tmp_path / "audit"))
    monkeypatch.setenv("AUDIT_FILE_FORMAT", "json")
    monkeypatch.setenv("AUDIT_FILE_ROTATION_STRATEGY", "time")


@pytest.fixture
def client(monkeypatch, mocker, tmp_path):
    """The app behind an always-active token, with the extraction never expected to run."""
    _set_baseline_env(monkeypatch, tmp_path)
    mocker.patch.object(
        auth,
        "validate_access_token",
        mocker.AsyncMock(return_value={"active": True, "sub": "caller-1", "aud": "intent_agent"}),
    )
    mocker.patch.object(IntentAgent, "run_extraction", mocker.AsyncMock())
    app, _agent = build_app(
        Settings(_env_file=None), llm=mocker.Mock(), schema_client=mocker.Mock(), policy_client=mocker.Mock()
    )
    with TestClient(app) as test_client:
        yield test_client


def _audit_records(tmp_path) -> list[dict]:
    return [
        json.loads(line)
        for audit_file in (tmp_path / "audit").glob("audit-intent_agent-*.json")
        for line in audit_file.read_text(encoding="utf-8").splitlines()
    ]


def _send_message(client: TestClient, metadata: dict):
    return client.post(
        "/",
        headers={"A2A-Version": "1.0", "Authorization": "Bearer good-token"},
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "SendMessage",
            "params": {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "Can I buy 50 shares of AAPL?"}],
                    "messageId": "integration-test-1",
                    "metadata": metadata,
                },
            },
        },
    )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"subject_id": "user-1"}, "subject_type is required"),
        ({"subject_type": "Customer"}, "subject_id is required"),
        ({"subject_type": "Customer", "subject_id": {"nested": "object"}}, "subject_id must be a string"),
    ],
    ids=["no-subject-type", "no-subject-id", "non-string"],
)
def test_invalid_metadata_is_a_jsonrpc_invalid_params_error(client, tmp_path, caplog, metadata, message):
    """Missing or malformed subject metadata is JSON-RPC -32602 with the existing message,
    audited once with that same message - refused before the SDK starts a task, so nothing is
    logged as a failed execution.
    """
    caplog.set_level(logging.ERROR)

    response = _send_message(client, metadata)

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == "1"
    assert "result" not in body
    assert body["error"]["code"] == _INVALID_PARAMS
    assert body["error"]["message"] == message
    IntentAgent.run_extraction.assert_not_awaited()
    (record,) = _audit_records(tmp_path)
    assert record["outcome"] == Outcome.ERROR
    assert record["message"] == message
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []
