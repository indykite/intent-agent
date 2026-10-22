"""The token audience check end to end; only introspection's HTTP transport is mocked, so the real check runs."""

from unittest.mock import AsyncMock

import httpx2
from langchain_core.messages import AIMessage
from starlette.status import HTTP_200_OK, HTTP_401_UNAUTHORIZED
from starlette.testclient import TestClient

from indykite.intent_agent.app import build_app
from indykite.intent_agent.config import Settings
from integration.schema_fixtures import RATING, STOCK_POLICIES, schema_with_stocks


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


def _mock_introspection_response(mocker, claims: dict) -> None:
    """Patch auth.py's httpx2.AsyncClient to answer introspection with the given claims."""
    client = mocker.MagicMock()
    client.__aenter__ = mocker.AsyncMock(return_value=client)
    client.__aexit__ = mocker.AsyncMock(return_value=False)
    client.post = mocker.AsyncMock(return_value=httpx2.Response(200, json=claims))
    mocker.patch("indykite.intent_agent.auth.httpx2.AsyncClient", return_value=client)


def _fake_schema_and_policy_clients(mocker):
    fake_schema_client = mocker.Mock(
        preload=mocker.Mock(),
        peek_cached=mocker.Mock(return_value=None),
        get_cached=mocker.Mock(return_value=schema_with_stocks()),
    )
    fake_policy_client = mocker.Mock(
        peek_cached=mocker.Mock(return_value=(False, None)),
        get_cached=mocker.Mock(return_value=(STOCK_POLICIES, False)),
    )
    return fake_schema_client, fake_policy_client


def _send_message(client: TestClient):
    return client.post(
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
                    "metadata": {"subject_type": "Customer", "subject_id": "test-subject"},
                },
            },
        },
    )


def test_rejects_token_scoped_to_a_different_audience(monkeypatch, mocker, tmp_path):
    """An active token whose `aud` doesn't name this agent gets a 401."""
    _set_baseline_env(monkeypatch, tmp_path)
    _mock_introspection_response(mocker, {"active": True, "aud": "some-other-service"})
    fake_schema_client, fake_policy_client = _fake_schema_and_policy_clients(mocker)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(side_effect=AssertionError("LLM must not be called")))

    settings = Settings(_env_file=None)
    app, _agent = build_app(settings, llm=fake_llm, schema_client=fake_schema_client, policy_client=fake_policy_client)
    client = TestClient(app)

    response = _send_message(client)

    assert response.status_code == HTTP_401_UNAUTHORIZED
    assert "Token is not valid for this agent" in response.text


def test_accepts_token_scoped_to_this_agent_end_to_end(monkeypatch, mocker, tmp_path):
    """A token whose `aud` is the default expected audience is accepted and extracted normally."""
    _set_baseline_env(monkeypatch, tmp_path)
    _mock_introspection_response(mocker, {"active": True, "aud": "intent_agent", "client_id": "agent-gateway"})
    fake_schema_client, fake_policy_client = _fake_schema_and_policy_clients(mocker)

    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"name": "activate_skill", "args": {"name": "extract-intent"}, "id": "call-1"}],
    )
    final_response = AIMessage(
        content=(
            '{"evaluations": '
            '[{"action": {"name": "buy"}, "resource": {"type": "Stock", "id": "AAPL"}, '
            '"context": {"parameters": {"amount": 50}, "filters": null, "qualifiers": null}}], '
            f"{RATING}}}"
        ),
        tool_calls=[],
    )
    mock_ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=fake_bound_llm))

    settings = Settings(_env_file=None)
    app, _agent = build_app(settings, llm=fake_llm, schema_client=fake_schema_client, policy_client=fake_policy_client)
    client = TestClient(app)

    response = _send_message(client)

    assert response.status_code == HTTP_200_OK
    task = response.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
