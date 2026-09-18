"""PRD: a request must be answered within 10s end to end. A hung LLM call must be cut off
by LLM_TIMEOUT_SECONDS and surface as the fixed error shape through the real A2A stack -
not just at the run_extraction level test_agent.py covers.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock

from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from indykite.intent_agent.app import build_app
from indykite.intent_agent.config import Settings
from indykite.intent_agent.errors import llm_timeout_error
from integration.schema_fixtures import load_schema_fixture

_PRD_REQUEST_BUDGET_SECONDS = 10.0
# Small so the test is quick, but the elapsed-time assertion is still well below the PRD
# budget: a request that ignored the timeout would sleep for the whole budget and beyond.
_LLM_TIMEOUT_SECONDS = 0.2
_MAX_ELAPSED_SECONDS = 2.0


def test_hung_llm_call_is_cut_off_within_request_budget(monkeypatch, mocker, tmp_path):
    """Drive a SendMessage request through the ASGI app against an LLM that never answers."""
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
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", str(_LLM_TIMEOUT_SECONDS))
    mocker.patch(
        "indykite.intent_agent.executor.auth.validate_access_token",
        return_value={"active": True, "sub": "test-subject"},
    )
    fake_schema_client = mocker.Mock(
        preload=mocker.Mock(),
        get_cached=mocker.Mock(return_value=load_schema_fixture()),
    )

    async def _hang(_messages):
        await asyncio.sleep(_PRD_REQUEST_BUDGET_SECONDS * 2)

    mock_ainvoke = AsyncMock(side_effect=_hang)
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    fake_llm = mocker.Mock(bind_tools=mocker.Mock(return_value=fake_bound_llm))

    settings = Settings(_env_file=None)
    app, _agent = build_app(settings, llm=fake_llm, schema_client=fake_schema_client)
    client = TestClient(app)

    started = time.perf_counter()
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
    elapsed = time.perf_counter() - started

    assert response.status_code == HTTP_200_OK
    result_text = response.json()["result"]["task"]["artifacts"][0]["parts"][0]["text"]
    assert json.loads(result_text) == llm_timeout_error("Can I buy 50 shares of AAPL?")
    assert mock_ainvoke.await_count == 1
    assert elapsed < _MAX_ELAPSED_SECONDS < _PRD_REQUEST_BUDGET_SECONDS
