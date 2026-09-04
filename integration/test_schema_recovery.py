import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import AsyncMock

import jwt
import pytest
from langchain_core.messages import AIMessage
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from indykite.intent_agent.agent import build_app
from indykite.intent_agent.config import Settings
from integration.schema_fixtures import schema_with_stocks

_SCHEMA_FIXTURE = schema_with_stocks()
_TOOL_CALL_ROUND_TRIPS = 2  # one activate_skill tool call, one final answer

# A real (unsigned, alg=none) JWT - SchemaClient requires any non-blank app-agent key to be a
# well-formed JWT with a 'sub' claim (see test_schema.py).
_APP_AGENT_KEY = jwt.encode({"sub": "test-app-agent"}, key="", algorithm="none")


class _SchemaHandler(BaseHTTPRequestHandler):
    """Serves 503 until `serving_schema` is flipped on, then the real schema fixture -
    a real HTTP server standing in for the IndyKite platform, not a mocked SchemaClient.
    """

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path != "/data-schema/v1" or not self.server.serving_schema:
            self.send_response(503)
            self.end_headers()
            return
        body = json.dumps(_SCHEMA_FIXTURE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def schema_server():
    """Start a real local HTTP server standing in for the platform's schema endpoint.

    Starts out failing (serving_schema=False); tests flip it on to simulate the schema
    becoming available partway through the process's lifetime.
    """
    server = HTTPServer(("127.0.0.1", 0), _SchemaHandler)
    server.serving_schema = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()


def _build_test_app(monkeypatch, mocker, schema_server, *, mock_ainvoke):
    """Wire up Settings/build_app pointed at the real local schema_server, with only the LLM
    and access-token introspection mocked (both are external services, same as
    test_send_message.py's existing integration test).
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("SUBJECT_TYPE", "Customer")
    monkeypatch.setenv("IK_APP_AGENT_KEY", _APP_AGENT_KEY)
    monkeypatch.setenv("INDYKITE_BASE_URL", f"http://{schema_server.server_address[0]}:{schema_server.server_port}")
    mocker.patch(
        "indykite.intent_agent.agent.auth.validate_access_token",
        return_value={"active": True, "sub": "test-subject"},
    )
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=fake_bound_llm))

    settings = Settings(_env_file=None)
    app, agent = build_app(settings, llm=fake_llm)  # schema_client=None -> real SchemaClient
    return TestClient(app), agent


def _send_message(client, text):
    """POST one SendMessage JSON-RPC request and return the extracted result text."""
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
                    "parts": [{"text": text}],
                    "messageId": "integration-test-1",
                },
            },
        },
    )
    assert response.status_code == HTTP_200_OK
    return response.json()["result"]["task"]["artifacts"][0]["parts"][0]["text"]


def test_schema_unavailable_at_cold_start_does_not_abort_startup(monkeypatch, mocker, schema_server):
    """build_app must not raise/exit when the real schema endpoint is failing at cold start -
    the process is expected to keep serving and retry the fetch per-request instead.
    """
    mock_ainvoke = AsyncMock()
    _client, agent = _build_test_app(monkeypatch, mocker, schema_server, mock_ainvoke=mock_ainvoke)

    assert agent.get_cached_schema() is None


def test_request_fails_cleanly_while_schema_still_unavailable(monkeypatch, mocker, schema_server):
    """A request made before the platform ever serves a schema must get the structured
    schema_unavailable error, without reaching the LLM.
    """
    mock_ainvoke = AsyncMock()
    client, _agent = _build_test_app(monkeypatch, mocker, schema_server, mock_ainvoke=mock_ainvoke)

    result_text = _send_message(client, "Can I buy 50 shares of AAPL?")

    result_data = json.loads(result_text)
    assert "error" in result_data
    assert "evaluations" not in result_data
    mock_ainvoke.assert_not_called()


def test_recovers_on_next_request_once_platform_publishes_schema(monkeypatch, mocker, schema_server):
    """Once the real platform starts serving the schema, the very next request on the same
    running agent (no restart) must pick it up and proceed through the LLM normally.
    """
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
    client, agent = _build_test_app(monkeypatch, mocker, schema_server, mock_ainvoke=mock_ainvoke)

    # First request: platform still schema-less.
    first_result = json.loads(_send_message(client, "Can I buy 50 shares of AAPL?"))
    assert "error" in first_result
    mock_ainvoke.assert_not_called()

    # Platform now publishes the schema - same process, no restart.
    schema_server.serving_schema = True

    result_text = _send_message(client, "Can I buy 50 shares of AAPL?")
    result_data = json.loads(result_text)
    assert result_data["evaluations"][0]["subject"] == {"type": "Customer", "id": "test-subject"}
    assert mock_ainvoke.await_count == _TOOL_CALL_ROUND_TRIPS
    assert agent.get_cached_schema() is not None
