import json
from typing import Any
from unittest.mock import AsyncMock

import httpx2
import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from indykite.intent_agent.app import build_app
from indykite.intent_agent.config import Settings
from integration.schema_fixtures import RATING, STOCK_POLICIES, schema_with_stocks


def _prepare(monkeypatch, mocker, tmp_path, **env):
    """Set the required settings plus `env`, accept any token, and return build_app's fake schema/policy clients."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
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
        "indykite.intent_agent.auth.validate_access_token",
        return_value={"active": True, "sub": "test-subject"},
    )
    # Fake clients: real ones would hit the network against the fake base URL.
    return {
        "schema_client": mocker.Mock(
            preload=mocker.Mock(),
            peek_cached=mocker.Mock(return_value=None),
            get_cached=mocker.Mock(return_value=schema_with_stocks()),
        ),
        "policy_client": mocker.Mock(
            peek_cached=mocker.Mock(return_value=(False, None)),
            get_cached=mocker.Mock(return_value=(STOCK_POLICIES, False)),
        ),
    }


def _send_buy_prompt(app):
    """POST the SendMessage every test here sends, as test-subject."""
    return TestClient(app).post(
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


def test_gemini_send_message_end_to_end(monkeypatch, mocker, tmp_path):
    """A SendMessage through the ASGI app (LLM mocked) returns the bound intent and writes one audit record."""
    clients = _prepare(monkeypatch, mocker, tmp_path, GEMINI_API_KEY="test-key")

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
    # The agent adds the prompt echo; null context keys must be dropped before reaching the caller.
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
    app, _agent = build_app(settings, llm=fake_llm, **clients)

    response = _send_buy_prompt(app)
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


_OPENAI_ANSWER = (
    '{"attack": false, "second_subject": null, "evaluations": '
    '[{"action": {"name": "buy"}, "resource": {"type": "Stock", "id": "AAPL"}, '
    '"context": {"parameters": {"amount": 50}, "filters": null, "qualifiers": null}}], '
    f"{RATING}}}"
)
_ACTIVATE_ARGUMENTS = '{"name": "extract-intent"}'
_OPENAI_ENV = {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"}


def _chat_completion(message: dict[str, Any], finish_reason: str) -> dict[str, Any]:
    """A Chat Completions API answer carrying one assistant message."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", **message}}],
    }


def _response(output_item: dict[str, Any]) -> dict[str, Any]:
    """A Responses API answer carrying one output item."""
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "test-model",
        "status": "completed",
        "output": [{"id": "item_1", "status": "completed", **output_item}],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


# Per OpenAI API: its path, the skill activation and final answer as it returns them, and where
# its request carries the structured-output format.
_OPENAI_APIS = {
    "chat_completions": (
        "/v1/chat/completions",
        [
            _chat_completion(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "activate_skill", "arguments": _ACTIVATE_ARGUMENTS},
                        }
                    ],
                },
                "tool_calls",
            ),
            _chat_completion({"content": _OPENAI_ANSWER}, "stop"),
        ],
        lambda body: body["response_format"]["json_schema"],
    ),
    "responses": (
        "/v1/responses",
        [
            _response(
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "activate_skill",
                    "arguments": _ACTIVATE_ARGUMENTS,
                }
            ),
            _response(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": _OPENAI_ANSWER, "annotations": []}],
                }
            ),
        ],
        lambda body: body["text"]["format"],
    ),
}


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_openai_send_message_end_to_end(monkeypatch, mocker, tmp_path, api):
    """With LLM_PROVIDER=openai, a SendMessage through the real OpenAI client (HTTP faked) returns the bound intent."""
    path, answers, format_of = _OPENAI_APIS[api]
    clients = _prepare(monkeypatch, mocker, tmp_path, **_OPENAI_ENV)
    requests: list[httpx2.Request] = []
    remaining = iter(answers)

    def openai_api(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(HTTP_200_OK, json=next(remaining))

    llm = ChatOpenAI(
        model="test-model",
        api_key="test-key",
        use_responses_api=api == "responses",
        http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(openai_api)),
    )
    app, _agent = build_app(Settings(_env_file=None), llm=llm, **clients)

    response = _send_buy_prompt(app)

    task = response.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert json.loads(task["artifacts"][0]["parts"][0]["text"]) == {
        "prompt": "Can I buy 50 shares of AAPL?",
        "evaluations": [
            {
                "action": {"name": "buy"},
                "resource": {"type": "Stock", "id": "AAPL"},
                "context": {"parameters": {"amount": 50}},
                "subject": {"type": "Customer", "id": "test-subject"},
            }
        ],
    }
    assert [request.url.path for request in requests] == [path, path]
    first, second = (json.loads(request.content) for request in requests)
    response_format = format_of(first)
    assert response_format["strict"] is False
    assert list(response_format["schema"]["properties"])[:2] == ["attack", "second_subject"]
    assert "temperature" not in first
    # Chat Completions nests a tool under "function"; the Responses API doesn't.
    assert [tool.get("function", tool)["strict"] for tool in first["tools"]] == [True, True, True]
    # The skill body went back to the model as the tool's result.
    assert "extract-intent" in json.dumps(second)
    assert len(json.dumps(second)) > len(json.dumps(first))

    record = json.loads(next((tmp_path / "audit").glob("audit-intent_agent-*.json")).read_text(encoding="utf-8"))
    assert (record["outcome"], record["llm_provider"], record["llm_model"]) == ("success", "openai", "test-model")


def test_openai_failure_is_reported_as_it_is_not_retried_into_a_timeout(monkeypatch, mocker, tmp_path):
    """A 429 from OpenAI, through the client the service builds itself: one call, answered with its status."""
    clients = _prepare(monkeypatch, mocker, tmp_path, **_OPENAI_ENV)
    requests: list[httpx2.Request] = []

    async def rate_limited(_transport, request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(429, json={"error": {"message": "Rate limit reached for sk-proj-a****0000"}})

    mocker.patch.object(httpx2.AsyncHTTPTransport, "handle_async_request", rate_limited)
    app, _agent = build_app(Settings(_env_file=None), **clients)

    response = _send_buy_prompt(app)

    assert len(requests) == 1
    assert response.json()["error"]["message"] == "openai call failed: OpenAIRateLimitError (status 429)"
    record = json.loads(next((tmp_path / "audit").glob("audit-intent_agent-*.json")).read_text(encoding="utf-8"))
    assert (record["outcome"], record["llm_provider"]) == ("error", "openai")
    assert record["message"] == "LlmProviderError: openai call failed: OpenAIRateLimitError (status 429)"
