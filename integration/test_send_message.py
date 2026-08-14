import json
from pathlib import Path
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from indykite.intent_agent.agent import build_app
from indykite.intent_agent.config import Settings

_SCHEMA_FIXTURE_PATH = Path(__file__).parent.parent / "test" / "fixtures" / "mock_ikg_schema.json"


def test_gemini_send_message_end_to_end(monkeypatch, mocker):
    """Drive a SendMessage JSON-RPC request through the ASGI app, LLM mocked."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("SUBJECT_TYPE", "Customer")
    monkeypatch.delenv("INDYKITE_BASE_URL", raising=False)
    monkeypatch.delenv("IK_APP_AGENT_KEY", raising=False)
    mocker.patch(
        "indykite.intent_agent.agent.auth.validate_access_token",
        return_value={"active": True, "sub": "test-subject"},
    )
    fake_schema_client = mocker.Mock(
        preload=mocker.Mock(),
        get_cached=mocker.Mock(return_value=json.loads(_SCHEMA_FIXTURE_PATH.read_text())),
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
        content=('{"prompt": "Can I buy 50 shares of AAPL?", "evaluations": [{"action": {"name": "buy"}}]}'),
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
