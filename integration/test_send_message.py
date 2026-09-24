import json
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from indykite.intent_agent.app import build_app
from indykite.intent_agent.config import Settings
from integration.schema_fixtures import schema_with_stocks


def test_gemini_send_message_end_to_end(monkeypatch, mocker, tmp_path):
    """Drive a SendMessage JSON-RPC request through the ASGI app, LLM mocked, and check the
    request left exactly one TDD 4.9 audit record behind through the real file sender.
    """
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
    mocker.patch(
        "indykite.intent_agent.executor.auth.validate_access_token",
        return_value={"active": True, "sub": "test-subject"},
    )
    fake_schema_client = mocker.Mock(
        preload=mocker.Mock(),
        peek_cached=mocker.Mock(return_value=None),
        get_cached=mocker.Mock(return_value=schema_with_stocks()),
    )
    # No policies (a successful fetch that found none) - this test is about the end-to-end
    # request path, not ENG-9671, and a real PolicyClient would otherwise hit the network
    # against this fake settings' base_url.
    fake_policy_client = mocker.Mock(
        peek_cached=mocker.Mock(return_value=(False, None)), get_cached=mocker.Mock(return_value=([], False))
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
    # The model's structured output is {evaluations} only - the prompt echo is added by the
    # agent. context.parameters / filters / qualifiers are passed through verbatim (TDD 9); a
    # null key (the structured output's nullable fields) is dropped rather than reaching the caller.
    final_response = AIMessage(
        content=(
            '{"evaluations": '
            '[{"action": {"name": "buy"}, "resource": {"type": "Stock", "id": "AAPL"}, '
            '"context": {"parameters": {"amount": 50}, "filters": null, "qualifiers": null}}]}'
        ),
        tool_calls=[],
    )
    mock_ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=fake_bound_llm))

    settings = Settings(_env_file=None)
    app, _agent = build_app(settings, llm=fake_llm, schema_client=fake_schema_client, policy_client=fake_policy_client)
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
                    "metadata": {"subject_type": "Customer", "subject_id": "test-subject"},
                },
            },
        },
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    task = body["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert "message" not in task["status"]
    assert task["history"][-1]["parts"][0]["text"] == "Intent extracted successfully."
    result_text = task["artifacts"][0]["parts"][0]["text"]
    assert '"buy"' in result_text
    assert mock_ainvoke.await_count == 2  # noqa: PLR2004 - one tool-call round, one final answer

    result_data = json.loads(result_text)
    assert list(result_data) == ["prompt", "evaluations"]
    assert result_data["prompt"] == "Can I buy 50 shares of AAPL?"
    assert result_data["evaluations"][0]["subject"] == {"type": "Customer", "id": "test-subject"}
    assert result_data["evaluations"][0]["context"] == {"parameters": {"amount": 50}}

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
