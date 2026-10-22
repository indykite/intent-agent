import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import AsyncMock

import jwt
import pytest
from langchain_core.messages import AIMessage
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from indykite.intent_agent.app import build_app
from indykite.intent_agent.config import Settings
from integration.schema_fixtures import RATING, STOCK_POLICIES, schema_with_stocks

_SCHEMA_FIXTURE = schema_with_stocks()
_TOOL_CALL_ROUND_TRIPS = 2  # one activate_skill tool call, one final answer

# SchemaClient requires the app-agent key to be a well-formed JWT with a 'sub' claim.
_APP_AGENT_KEY = jwt.encode({"sub": "test-app-agent"}, key="", algorithm="none")


class _SchemaHandler(BaseHTTPRequestHandler):
    """Stands in for the platform: 503 until `serving_schema` is set, then the schema and policies."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        documents = {"/data-schema/v1": _SCHEMA_FIXTURE, "/access/v1/policies": {"results": STOCK_POLICIES}}
        document = documents.get(self.path.split("?", 1)[0])
        if document is None or not self.server.serving_schema:
            self.send_response(503)
            self.end_headers()
            return
        body = json.dumps(document).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def schema_server():
    """A real local platform stand-in that starts out failing (serving_schema=False)."""
    server = HTTPServer(("127.0.0.1", 0), _SchemaHandler)
    server.serving_schema = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()


def _build_test_app(monkeypatch, mocker, schema_server, tmp_path, *, mock_ainvoke):
    """build_app against the local schema_server, with only the LLM and token introspection mocked."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("IK_APP_AGENT_KEY", _APP_AGENT_KEY)
    monkeypatch.setenv("INDYKITE_BASE_URL", f"http://{schema_server.server_address[0]}:{schema_server.server_port}")
    monkeypatch.setenv("AUDIT_DELIVERY", "file")
    monkeypatch.setenv("AUDIT_FILE_STORAGE_PATH", str(tmp_path / "audit"))
    monkeypatch.setenv("AUDIT_FILE_FORMAT", "json")
    monkeypatch.setenv("AUDIT_FILE_ROTATION_STRATEGY", "time")
    mocker.patch(
        "indykite.intent_agent.auth.validate_access_token",
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
                    "metadata": {"subject_type": "Customer", "subject_id": "test-subject"},
                },
            },
        },
    )
    assert response.status_code == HTTP_200_OK
    return response.json()["result"]["task"]["artifacts"][0]["parts"][0]["text"]


def test_schema_unavailable_at_cold_start_does_not_abort_startup(monkeypatch, mocker, schema_server, tmp_path):
    mock_ainvoke = AsyncMock()
    _client, agent = _build_test_app(monkeypatch, mocker, schema_server, tmp_path, mock_ainvoke=mock_ainvoke)

    assert agent.get_cached_schema() is None


def test_request_fails_cleanly_while_schema_still_unavailable(monkeypatch, mocker, schema_server, tmp_path):
    """Before any schema is served a request gets the schema_unavailable error without reaching the LLM."""
    mock_ainvoke = AsyncMock()
    client, _agent = _build_test_app(monkeypatch, mocker, schema_server, tmp_path, mock_ainvoke=mock_ainvoke)

    result_text = _send_message(client, "Can I buy 50 shares of AAPL?")

    result_data = json.loads(result_text)
    assert "error" in result_data
    assert "evaluations" not in result_data
    mock_ainvoke.assert_not_called()


def test_recovers_on_next_request_once_platform_publishes_schema(monkeypatch, mocker, schema_server, tmp_path):
    """Once the schema is served, the next request on the same process picks it up, no restart."""
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"name": "activate_skill", "args": {"name": "extract-intent"}, "id": "call-1"}],
    )
    final_response = AIMessage(
        content=(
            '{"prompt": "Can I buy 50 shares of AAPL?", "evaluations": '
            '[{"action": {"name": "buy"}, "resource": {"type": "Stock", "id": "AAPL"}}], '
            f"{RATING}}}"
        ),
        tool_calls=[],
    )
    mock_ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])
    client, agent = _build_test_app(monkeypatch, mocker, schema_server, tmp_path, mock_ainvoke=mock_ainvoke)

    first_result = json.loads(_send_message(client, "Can I buy 50 shares of AAPL?"))
    assert "error" in first_result
    mock_ainvoke.assert_not_called()

    schema_server.serving_schema = True

    result_text = _send_message(client, "Can I buy 50 shares of AAPL?")
    result_data = json.loads(result_text)
    assert result_data["evaluations"][0]["subject"] == {"type": "Customer", "id": "test-subject"}
    assert mock_ainvoke.await_count == _TOOL_CALL_ROUND_TRIPS
    assert agent.get_cached_schema() is not None
