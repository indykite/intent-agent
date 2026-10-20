"""Every A2A JSON-RPC method is authenticated, and only SendMessage is served."""

import asyncio
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from starlette.status import HTTP_200_OK, HTTP_401_UNAUTHORIZED
from starlette.testclient import TestClient

from indykite.a2a.task_store import EvictingTaskStore
from indykite.intent_agent import auth
from indykite.intent_agent.agent import ExtractionResult, IntentAgent
from indykite.intent_agent.app import build_app
from indykite.intent_agent.audit.message import Outcome
from indykite.intent_agent.config import Settings

_A2A_HEADERS = {"A2A-Version": "1.0"}
_CALLERS = {"alice-token": "alice", "bob-token": "bob"}
_INVALID_PARAMS = -32602
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


def _send_message(client: TestClient, token: str | None, *, message_id: str = "integration-test-1", task_id: str = ""):
    message = {
        "role": "ROLE_USER",
        "parts": [{"text": "Can I buy 50 shares of AAPL?"}],
        "messageId": message_id,
        "metadata": {"subject_type": "Customer", "subject_id": "test-subject"},
    }
    if task_id:
        message["taskId"] = task_id
    return _rpc(client, "SendMessage", {"message": message}, token)


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


@pytest.mark.parametrize("token", ["alice-token", "bob-token"], ids=["same-token", "other-token"])
def test_a_message_naming_a_task_is_refused(client, mocker, tmp_path, token):
    """A SendMessage naming a `taskId` - here a still-running task - is InvalidParams (-32602) and
    audited, whoever sends it: the task is never loaded, continued or returned.
    """
    release = threading.Event()
    saved_task_ids = queue.Queue()
    save = EvictingTaskStore.save

    async def _record_save(store, task, context):
        await save(store, task, context)
        saved_task_ids.put(task.id)

    async def _held_extraction(*_args):
        await asyncio.to_thread(release.wait, 5)
        return ExtractionResult(text="{}", outcome=Outcome.SUCCESS)

    mocker.patch.object(EvictingTaskStore, "save", _record_save)
    mocker.patch.object(IntentAgent, "run_extraction", _held_extraction)
    safety_release = threading.Timer(5, release.set)
    safety_release.start()

    with ThreadPoolExecutor(max_workers=1) as pool:
        alice = pool.submit(_send_message, client, "alice-token")
        task_id = saved_task_ids.get(timeout=5)
        second = _send_message(client, token, message_id="integration-test-2", task_id=task_id)
        release.set()
        alice_task = alice.result(timeout=5).json()["result"]["task"]
    safety_release.cancel()

    assert second.status_code == HTTP_200_OK
    assert second.json()["error"]["code"] == _INVALID_PARAMS
    assert second.json()["error"]["message"] == "taskId is not supported"
    assert "result" not in second.json()
    assert alice_task["id"] == task_id
    history_ids = [message["messageId"] for message in alice_task["history"]]
    assert "integration-test-1" in history_ids
    assert "integration-test-2" not in history_ids
    refused = [record for record in _audit_records(tmp_path) if record["message"] == "taskId is not supported"]
    assert len(refused) == 1
    assert refused[0]["outcome"] == "error"
    assert refused[0]["prompt"] == "Can I buy 50 shares of AAPL?"


def test_agent_card_stays_public(client):
    """The agent card needs no token, and tells the client it needs one for everything else: a
    Bearer scheme matching the challenge the 401 carries.
    """
    response = client.get(AGENT_CARD_WELL_KNOWN_PATH)

    assert response.status_code == HTTP_200_OK
    card = response.json()
    assert card["name"]
    (name,) = card["securitySchemes"]
    advertised = card["securitySchemes"][name]["httpAuthSecurityScheme"]["scheme"]
    assert "'intent_agent'" in card["securitySchemes"][name]["httpAuthSecurityScheme"]["description"]
    assert card["securityRequirements"] == [{"schemes": {name: {}}}]
    challenge = _rpc(client, "SendMessage", {}).headers["WWW-Authenticate"]
    assert challenge.split()[0] == advertised == "Bearer"


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
