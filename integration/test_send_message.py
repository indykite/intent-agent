import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from starlette.status import HTTP_200_OK, HTTP_401_UNAUTHORIZED
from starlette.testclient import TestClient

from indykite.intent_agent.agent import build_app
from indykite.intent_agent.config import Settings
from integration.schema_fixtures import schema_with_stocks

_SEND_MESSAGE_BODY = {
    "jsonrpc": "2.0",
    "id": "1",
    "method": "SendMessage",
    "params": {
        "message": {
            "role": "ROLE_USER",
            "parts": [{"text": "Can I buy 50 shares of AAPL?"}],
            "messageId": "integration-test-1",
        },
    },
}


def _set_env(monkeypatch, tmp_path, *, its: bool = False):
    """Every required Settings env var, file audit delivery under tmp_path, and the ITS_* triple when `its` is set."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("SUBJECT_TYPE", "Customer")
    monkeypatch.setenv("INDYKITE_BASE_URL", "https://ikg.example.com")
    monkeypatch.setenv("IK_APP_AGENT_KEY", "test-app-agent-key")
    monkeypatch.setenv("AUDIT_DELIVERY", "file")
    monkeypatch.setenv("AUDIT_FILE_STORAGE_PATH", str(tmp_path / "audit"))
    monkeypatch.setenv("AUDIT_FILE_FORMAT", "json")
    monkeypatch.setenv("AUDIT_FILE_ROTATION_STRATEGY", "time")
    for env_var in ("ITS_INTROSPECT_URL", "ITS_CLIENT_ID", "ITS_CLIENT_SECRET"):
        monkeypatch.delenv(env_var, raising=False)
    if its:
        monkeypatch.setenv("ITS_INTROSPECT_URL", "https://its.example.com/oauth2/introspect")
        monkeypatch.setenv("ITS_CLIENT_ID", "its-client-id")
        monkeypatch.setenv("ITS_CLIENT_SECRET", "its-client-secret")


def _client_with_mocked_llm(mocker):
    """A TestClient over build_app with the schema and the LLM's two answers mocked."""
    fake_schema_client = mocker.Mock(
        preload=mocker.Mock(),
        get_cached=mocker.Mock(return_value=schema_with_stocks()),
    )
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"name": "activate_skill", "args": {"name": "extract-intent"}, "id": "call-1"}],
    )
    final_response = AIMessage(
        content=(
            '{"prompt": "Can I buy 50 shares of AAPL?", "evaluations": '
            '[{"action": {"name": "buy"}, "resource": {"type": "Stock", "id": "AAPL"}}]}'
        ),
        tool_calls=[],
    )
    mock_ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=fake_bound_llm))
    app, _agent = build_app(Settings(_env_file=None), llm=fake_llm, schema_client=fake_schema_client)
    return TestClient(app), mock_ainvoke


def test_dual_token_send_message_end_to_end(monkeypatch, mocker, tmp_path):
    """With ITS configured, Authorization plus X-IK-Token (raw, no Bearer) drive a SendMessage
    request through the ASGI app: both validators see their header's token, and the subject
    still comes from the access token.
    """
    _set_env(monkeypatch, tmp_path, its=True)
    validate_access = mocker.patch(
        "indykite.intent_agent.agent.auth.validate_access_token",
        return_value={"active": True, "sub": "test-subject"},
    )
    validate_ik = mocker.patch(
        "indykite.intent_agent.agent.auth.validate_ik_token",
        return_value={"active": True, "sub": "test-subject", "act": {"sub": "orchestrator", "type": "Agent"}},
    )
    client, _mock_ainvoke = _client_with_mocked_llm(mocker)

    response = client.post(
        "/",
        headers={"A2A-Version": "1.0", "Authorization": "Bearer test-token", "X-IK-Token": "test-ik-token"},
        json=_SEND_MESSAGE_BODY,
    )

    assert response.status_code == HTTP_200_OK
    assert validate_access.call_args.args == ("test-token",)
    assert validate_ik.call_args.args == ("test-ik-token",)
    result_text = response.json()["result"]["task"]["artifacts"][0]["parts"][0]["text"]
    assert json.loads(result_text)["evaluations"][0]["subject"] == {"type": "Customer", "id": "test-subject"}


@pytest.mark.parametrize(
    ("headers", "detail"),
    [
        pytest.param({"Authorization": "Bearer test-token"}, "X-IK-Token required", id="missing_ik_token"),
        pytest.param({"X-IK-Token": "test-ik-token"}, "Authorization required", id="missing_access_token"),
    ],
)
def test_dual_token_send_message_rejects_missing_token(monkeypatch, mocker, tmp_path, headers, detail):
    """With ITS configured, either token missing is a 401 at the HTTP layer (TDD 3.1)."""
    _set_env(monkeypatch, tmp_path, its=True)
    mocker.patch(
        "indykite.intent_agent.agent.auth.validate_access_token",
        return_value={"active": True, "sub": "test-subject"},
    )
    client, mock_ainvoke = _client_with_mocked_llm(mocker)

    response = client.post("/", headers={"A2A-Version": "1.0", **headers}, json=_SEND_MESSAGE_BODY)

    assert response.status_code == HTTP_401_UNAUTHORIZED
    assert detail in response.text
    assert mock_ainvoke.await_count == 0


def test_gemini_send_message_end_to_end(monkeypatch, mocker, tmp_path):
    """Drive a SendMessage JSON-RPC request through the ASGI app, LLM mocked, and check the
    request left exactly one TDD 4.9 audit record behind through the real file sender.
    """
    _set_env(monkeypatch, tmp_path)
    mocker.patch(
        "indykite.intent_agent.agent.auth.validate_access_token",
        return_value={"active": True, "sub": "test-subject"},
    )
    fake_schema_client = mocker.Mock(
        preload=mocker.Mock(),
        get_cached=mocker.Mock(return_value=schema_with_stocks()),
    )

    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "activate_skill",
                "args": {"name": "extract-intent"},
                "id": "call-1",
            }
        ],
    )
    final_response = AIMessage(
        content=(
            '{"prompt": "Can I buy 50 shares of AAPL?", "evaluations": '
            '[{"action": {"name": "buy"}, "resource": {"type": "Stock", "id": "AAPL"}}]}'
        ),
        tool_calls=[],
    )
    mock_ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=fake_bound_llm))

    settings = Settings(_env_file=None)
    app, _agent = build_app(settings, llm=fake_llm, schema_client=fake_schema_client)
    client = TestClient(app)

    response = client.post(
        "/",
        headers={"A2A-Version": "1.0", "Authorization": "Bearer test-token"},
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "SendMessage",
            "params": {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "Can I buy 50 shares of AAPL?"}],
                    "messageId": "integration-test-1",
                },
            },
        },
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    result_text = body["result"]["task"]["artifacts"][0]["parts"][0]["text"]
    assert '"buy"' in result_text
    assert mock_ainvoke.await_count == 2  # noqa: PLR2004 - one tool-call round, one final answer

    result_data = json.loads(result_text)
    assert result_data["evaluations"][0]["subject"] == {"type": "Customer", "id": "test-subject"}

    audit_files = list((tmp_path / "audit").glob("audit-intent_agent-*.json"))
    assert len(audit_files) == 1
    audit_lines = audit_files[0].read_text(encoding="utf-8").splitlines()
    assert len(audit_lines) == 1
    record = json.loads(audit_lines[0])
    assert record["outcome"] == "success"
    assert record["subject_id"] == "test-subject"
    assert record["subject_type"] == "Customer"
    assert record["prompt"] == "Can I buy 50 shares of AAPL?"
    assert record["evaluations"] == result_data["evaluations"]
    assert record["llm_provider"] == "gemini"
    assert record["message"] == ""
    assert record["latency_ms"] >= 0
