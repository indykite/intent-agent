"""Redis failure-mode tests against the full ASGI stack: a refused or blackholed cache backend
must not fail the request, and two environments sharing one real Redis must stay isolated
(beyond test_redis_isolation.py's import-only check). Needs a real Redis reachable for the probe.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import AsyncMock

import jwt
import pytest
import redis
from langchain_core.messages import AIMessage
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from indykite.intent_agent.app import build_app
from indykite.intent_agent.config import Settings
from indykite.intent_agent.schema import CacheConfig, IkConnection, SchemaClient, env_scope
from indykite.redis.storage import build_redis_storage_backend
from integration.conftest import REAL_REDIS_URL, real_redis_reachable
from integration.schema_fixtures import RATING, STOCK_POLICIES, schema_with_stocks

pytestmark = pytest.mark.skipif(
    not real_redis_reachable(),
    reason=f"no real Redis reachable at {REAL_REDIS_URL} - see module docstring",
)

_SCHEMA_FIXTURE = schema_with_stocks()
_REQUEST_TIMEOUT_SECONDS = 5.0
_REDIS_SOCKET_TIMEOUT_SECONDS = 1.0
# Comfortably above _REDIS_SOCKET_TIMEOUT_SECONDS (the blackholed connect must be cut off by
# it), comfortably below _REQUEST_TIMEOUT_SECONDS - proves the stall never ate the whole budget.
_MAX_ELAPSED_SECONDS = 3.0
# A non-routable address: connections to it hang until the OS/kernel gives up, unlike a refused
# connection (instant RST) - the actual "blackholed" case, not just "nothing listening".
_BLACKHOLED_REDIS_URL = "redis://10.255.255.1:6379"
# A real port nothing listens on - connection refused outright, fails fast.
_REFUSED_REDIS_URL = "redis://localhost:1"
_SCHEMA_REFRESH_TTL_SECONDS = 300.0


def _fake_jwt(sub: str) -> str:
    """A real (unsigned, alg=none) JWT - SchemaClient/PolicyClient require any non-blank
    app-agent key to be a well-formed JWT with a 'sub' claim.
    """
    return jwt.encode({"sub": sub}, key="", algorithm="none")


class _PlatformHandler(BaseHTTPRequestHandler):
    """A real local HTTP server standing in for the IndyKite platform - always answers, so the
    only variable in these tests is Redis's own reachability.
    """

    def log_message(self, *args):
        pass

    def do_GET(self):
        documents = {"/data-schema/v1": _SCHEMA_FIXTURE, "/access/v1/policies": {"results": STOCK_POLICIES}}
        document = documents.get(self.path.split("?", 1)[0])
        body = json.dumps(document).encode()
        self.send_response(HTTP_200_OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def platform_server():
    """Start the real local platform stand-in server."""
    server = HTTPServer(("127.0.0.1", 0), _PlatformHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()


def _configure_settings_env(monkeypatch, tmp_path, platform_server, *, redis_url: str) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("IK_APP_AGENT_KEY", _fake_jwt("redis-resilience-test"))
    monkeypatch.setenv("INDYKITE_BASE_URL", f"http://{platform_server.server_address[0]}:{platform_server.server_port}")
    monkeypatch.setenv("AUDIT_DELIVERY", "file")
    monkeypatch.setenv("AUDIT_FILE_STORAGE_PATH", str(tmp_path / "audit"))
    monkeypatch.setenv("AUDIT_FILE_FORMAT", "json")
    monkeypatch.setenv("AUDIT_FILE_ROTATION_STRATEGY", "time")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", str(_REQUEST_TIMEOUT_SECONDS))
    monkeypatch.setenv("CACHE_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("REDIS_SOCKET_TIMEOUT_SECONDS", str(_REDIS_SOCKET_TIMEOUT_SECONDS))


def _fake_buy_aapl_llm(mocker):
    """An LLM mock that activates extract-intent then answers "buy AAPL" - the same shape
    test_schema_recovery.py/test_send_message.py already use.
    """
    tool_call_response = AIMessage(
        content="", tool_calls=[{"name": "activate_skill", "args": {"name": "extract-intent"}, "id": "call-1"}]
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
    fake_bound_llm = mocker.Mock(ainvoke=mock_ainvoke)
    return mocker.Mock(bind_tools=mocker.Mock(return_value=fake_bound_llm))


def _send_message(client: TestClient, text: str) -> dict:
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
    return response.json()["result"]["task"]


def _one_audit_record(tmp_path) -> dict:
    audit_files = list((tmp_path / "audit").glob("audit-intent_agent-*.json"))
    assert len(audit_files) == 1
    audit_lines = audit_files[0].read_text(encoding="utf-8").splitlines()
    assert len(audit_lines) == 1
    return json.loads(audit_lines[0])


def test_refused_redis_still_succeeds_and_audits_normally(monkeypatch, mocker, tmp_path, platform_server):
    """A refused Redis connection must not fail the request - clients fall back to a direct
    platform fetch - and the audit record must still read a clean "success", not expose Redis.
    """
    _configure_settings_env(monkeypatch, tmp_path, platform_server, redis_url=_REFUSED_REDIS_URL)
    mocker.patch(
        "indykite.intent_agent.executor.auth.validate_access_token",
        return_value={"active": True, "sub": "test-subject"},
    )
    settings = Settings(_env_file=None)
    app, _agent = build_app(settings, llm=_fake_buy_aapl_llm(mocker))
    client = TestClient(app)

    task = _send_message(client, "Can I buy 50 shares of AAPL?")

    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    result_data = json.loads(task["artifacts"][0]["parts"][0]["text"])
    assert result_data["evaluations"][0]["subject"] == {"type": "Customer", "id": "test-subject"}

    record = _one_audit_record(tmp_path)
    assert record["outcome"] == "success"
    assert record["subject_id"] == "test-subject"


def test_blackholed_redis_stays_within_the_request_budget(monkeypatch, mocker, tmp_path, platform_server):
    """A blackholed Redis (connects but never answers) must not eat the request budget:
    REDIS_SOCKET_TIMEOUT_SECONDS bounds the stall, then the same fallback lets it succeed.
    """
    _configure_settings_env(monkeypatch, tmp_path, platform_server, redis_url=_BLACKHOLED_REDIS_URL)
    mocker.patch(
        "indykite.intent_agent.executor.auth.validate_access_token",
        return_value={"active": True, "sub": "test-subject"},
    )
    settings = Settings(_env_file=None)
    app, _agent = build_app(settings, llm=_fake_buy_aapl_llm(mocker))
    client = TestClient(app)

    started = time.perf_counter()
    task = _send_message(client, "Can I buy 50 shares of AAPL?")
    elapsed = time.perf_counter() - started

    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert elapsed < _MAX_ELAPSED_SECONDS < _REQUEST_TIMEOUT_SECONDS


@pytest.fixture
def two_environments_on_real_redis():
    """Two SchemaClients for different IndyKite environments sharing one real Redis; cleans up
    both environments' keys afterward so repeated runs don't leave test data behind.
    """
    app_agent_key_a = _fake_jwt("env-a")
    app_agent_key_b = _fake_jwt("env-b")
    backend_a = build_redis_storage_backend(REAL_REDIS_URL, socket_timeout_seconds=2.0)
    backend_b = build_redis_storage_backend(REAL_REDIS_URL, socket_timeout_seconds=2.0)
    client_a = SchemaClient(
        IkConnection("https://a.example.com", app_agent_key_a),
        cache_config=CacheConfig(
            cache=backend_a, refresh_ttl_seconds=_SCHEMA_REFRESH_TTL_SECONDS, executor=ThreadPoolExecutor(max_workers=1)
        ),
    )
    client_b = SchemaClient(
        IkConnection("https://b.example.com", app_agent_key_b),
        cache_config=CacheConfig(
            cache=backend_b, refresh_ttl_seconds=_SCHEMA_REFRESH_TTL_SECONDS, executor=ThreadPoolExecutor(max_workers=1)
        ),
    )
    try:
        yield client_a, client_b
    finally:
        for key_scope in (env_scope(app_agent_key_a), env_scope(app_agent_key_b)):
            backend_a._client.delete(f"ikg:{key_scope}:schema", f"ikg:{key_scope}:schema:fresh")  # noqa: SLF001
        backend_a.close()
        backend_b.close()


def test_two_environments_on_one_real_redis_only_read_their_own_keys(mocker, two_environments_on_real_redis):
    """Two environments sharing one real Redis must never read each other's schema - proves the
    ikg:<gid>:* key scoping holds against an actual server, not just fakeredis.
    """
    client_a, client_b = two_environments_on_real_redis
    schema_a = {"graph": {"nodes": {"OnlyInA": {}}, "edges": []}}
    schema_b = {"graph": {"nodes": {"OnlyInB": {}}, "edges": []}}
    mocker.patch.object(SchemaClient, "load", side_effect=[schema_a, schema_b])

    client_a.preload()
    client_b.preload()

    assert client_a.get_cached() == schema_a
    assert client_b.get_cached() == schema_b


def test_one_transient_connection_blip_is_retried_not_surfaced(mocker):
    """storage.py's _CONNECTION_RETRY must absorb one dropped-connection error transparently
    rather than falling back to the cache-failure path; simulated by making send_command raise once.
    """
    backend = build_redis_storage_backend(REAL_REDIS_URL, socket_timeout_seconds=2.0)
    key = "ikg:connection-retry-test:probe"
    orig_send_command = redis.connection.Connection.send_command
    call_count = 0

    def flaky_send_command(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            message = "simulated transient blip"
            raise redis.exceptions.ConnectionError(message)
        return orig_send_command(self, *args, **kwargs)

    mocker.patch.object(redis.connection.Connection, "send_command", flaky_send_command)
    try:
        backend.set(key, b"value")
        assert backend.get(key) == b"value"
    finally:
        backend.delete(key)
        backend.close()
    assert call_count > 1
