"""Every A2A JSON-RPC method is authenticated, and only SendMessage is served."""

import asyncio
import json

import pytest
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from starlette.status import HTTP_200_OK, HTTP_401_UNAUTHORIZED
from starlette.testclient import TestClient

from indykite.intent_agent import auth
from indykite.intent_agent.agent import ExtractionResult, IntentAgent
from indykite.intent_agent.app import build_app
from indykite.intent_agent.audit.message import Outcome
from indykite.intent_agent.config import Settings

_A2A_HEADERS = {"A2A-Version": "1.0"}
_CALLERS = {"alice-token": "alice", "bob-token": "bob"}
_UNSUPPORTED_OPERATION = -32004


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


async def _introspect(token, _config):
    """Two known callers' tokens are active; anything else is an inactive token."""
    if token in _CALLERS:
        return {"active": True, "sub": _CALLERS[token], "aud": "intent_agent"}
    message = "Token is not active"
    raise auth.TokenValidationError(message, auth.TokenFailure.CALLER)


@pytest.fixture
def client(monkeypatch, mocker, tmp_path):
    """The app with its extraction stubbed out."""
    _set_baseline_env(monkeypatch, tmp_path)
    mocker.patch.object(auth, "validate_access_token", _introspect)
    mocker.patch.object(
        IntentAgent,
        "run_extraction",
        mocker.AsyncMock(return_value=ExtractionResult(text="{}", outcome=Outcome.SUCCESS)),
    )
    app, _agent = build_app(
        Settings(_env_file=None), llm=mocker.Mock(), schema_client=mocker.Mock(), policy_client=mocker.Mock()
    )
    with TestClient(app) as test_client:
        yield test_client


def _rpc(client: TestClient, method: str, params: dict, token: str | None = None):
    headers = dict(_A2A_HEADERS)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return client.post("/", headers=headers, json={"jsonrpc": "2.0", "id": "1", "method": method, "params": params})


def _send_message(client: TestClient, token: str | None):
    return _rpc(
        client,
        "SendMessage",
        {
            "message": {
                "role": "ROLE_USER",
                "parts": [{"text": "Can I buy 50 shares of AAPL?"}],
                "messageId": "integration-test-1",
                "metadata": {"subject_type": "Customer", "subject_id": "test-subject"},
            },
        },
        token,
    )


def _audit_records(tmp_path) -> list[dict]:
    return [
        json.loads(line)
        for audit_file in (tmp_path / "audit").glob("audit-intent_agent-*.json")
        for line in audit_file.read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.parametrize("token", [None, "junk-token"], ids=["no-token", "invalid-token"])
@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("SendMessage", None),
        ("GetTask", {"id": "some-task"}),
        ("ListTasks", {}),
        ("CancelTask", {"id": "some-task"}),
    ],
)
def test_every_method_needs_a_valid_token(client, tmp_path, method, params, token):
    """Every method is a 401 without a valid token, and each refusal is audited once."""
    response = _send_message(client, token) if params is None else _rpc(client, method, params, token)

    assert response.status_code == HTTP_401_UNAUTHORIZED
    assert response.headers["WWW-Authenticate"].startswith("Bearer")
    records = _audit_records(tmp_path)
    assert len(records) == 1
    assert records[0]["outcome"] == "error"
    assert records[0]["message"] == response.text
    if method == "SendMessage":
        assert records[0]["prompt"] == "Can I buy 50 shares of AAPL?"


@pytest.mark.parametrize("token", ["alice-token", "bob-token"], ids=["task-owner", "other-caller"])
@pytest.mark.parametrize("method", ["GetTask", "ListTasks", "CancelTask"])
def test_task_methods_are_unsupported(client, method, token):
    """Only SendMessage is served: a valid token gets UnsupportedOperation for the task methods."""
    sent = _send_message(client, "alice-token")
    assert sent.status_code == HTTP_200_OK
    task_id = sent.json()["result"]["task"]["id"]

    response = _rpc(client, method, {} if method == "ListTasks" else {"id": task_id}, token)

    assert response.status_code == HTTP_200_OK
    assert response.json()["error"]["code"] == _UNSUPPORTED_OPERATION
    assert "result" not in response.json()


def test_agent_card_stays_public(client):
    """The agent card needs no token."""
    response = client.get(AGENT_CARD_WELL_KNOWN_PATH)

    assert response.status_code == HTTP_200_OK
    assert response.json()["name"]


@pytest.mark.parametrize("token", ["alice-token", "junk-token"], ids=["accepted", "refused"])
def test_audit_latency_includes_introspection(client, mocker, tmp_path, token):
    """`latency_ms` covers the introspection call, whether the token is accepted or refused."""
    delay_s = 0.3

    async def _slow_introspect(token, config):
        await asyncio.sleep(delay_s)
        return await _introspect(token, config)

    mocker.patch.object(auth, "validate_access_token", _slow_introspect)

    _send_message(client, token)

    (record,) = _audit_records(tmp_path)
    assert record["latency_ms"] >= delay_s * 1000
