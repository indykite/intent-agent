"""Tests for app.py's _build_task_store_and_queue_manager: the A2A_TASK_STORE_BACKEND
memory/redis toggle build_app() wires the task store/queue manager from.
"""

import jwt
from a2a.server.tasks import InMemoryTaskStore
from fakeredis import FakeAsyncRedis
from starlette.testclient import TestClient

from indykite.a2a.redis.queue_manager import RedisQueueManager
from indykite.a2a.redis.task_store import RedisTaskStore
from indykite.intent_agent.app import _build_task_store_and_queue_manager, build_app
from indykite.intent_agent.config import Settings

_APP_AGENT_KEY = jwt.encode({"sub": "test-app-agent"}, key="", algorithm="none")


def _set_baseline_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("IK_APP_AGENT_KEY", _APP_AGENT_KEY)
    monkeypatch.setenv("INDYKITE_BASE_URL", "https://ikg.example.com")
    monkeypatch.setenv("AUDIT_DELIVERY", "file")
    monkeypatch.setenv("AUDIT_FILE_STORAGE_PATH", "./.private/audit-never-created")
    monkeypatch.setenv("AUDIT_FILE_FORMAT", "json")
    monkeypatch.setenv("AUDIT_FILE_ROTATION_STRATEGY", "time")


def test_uses_in_memory_task_store_by_default(monkeypatch):
    """A2A_TASK_STORE_BACKEND defaults to "memory" - the a2a-sdk's own InMemoryTaskStore, no
    Redis client opened, and no queue manager.
    """
    _set_baseline_env(monkeypatch)
    settings = Settings(_env_file=None)

    task_store, queue_manager = _build_task_store_and_queue_manager(settings)

    assert isinstance(task_store, InMemoryTaskStore)
    assert queue_manager is None


def test_uses_redis_task_store_and_queue_manager_when_configured(monkeypatch):
    """A2A_TASK_STORE_BACKEND=redis must wire RedisTaskStore/RedisQueueManager, sharing one
    Redis client and env_id between them.
    """
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("A2A_TASK_STORE_BACKEND", "redis")
    monkeypatch.setenv("A2A_TASK_STORE_URL", "redis://redis.example.com:6379")
    settings = Settings(_env_file=None)

    task_store, queue_manager = _build_task_store_and_queue_manager(settings)

    assert isinstance(task_store, RedisTaskStore)
    assert isinstance(queue_manager, RedisQueueManager)
    assert task_store._env_id == queue_manager._env_id
    assert task_store._backend._client is queue_manager._client


def test_build_app_closes_the_shared_redis_client_on_shutdown(monkeypatch, mocker):
    """The app's lifespan must close the shared build_redis_client() client (task store +
    queue manager).
    """
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("A2A_TASK_STORE_BACKEND", "redis")
    monkeypatch.setenv("A2A_TASK_STORE_URL", "redis://redis.example.com:6379")
    client = FakeAsyncRedis()
    client.aclose = mocker.AsyncMock(wraps=client.aclose)
    mocker.patch("indykite.intent_agent.app.build_redis_client", return_value=client)
    fake_schema_client = mocker.Mock(preload=mocker.Mock(), aclose=mocker.AsyncMock())
    fake_policy_client = mocker.Mock(aclose=mocker.AsyncMock())
    settings = Settings(_env_file=None)
    app, _agent = build_app(settings, schema_client=fake_schema_client, policy_client=fake_policy_client)

    with TestClient(app):
        client.aclose.assert_not_awaited()
    client.aclose.assert_awaited_once()


def test_build_app_in_memory_mode_has_no_redis_client_to_close(monkeypatch, mocker):
    """The default (memory) backend must not construct a Redis client at all."""
    _set_baseline_env(monkeypatch)
    build_redis_client = mocker.patch("indykite.intent_agent.app.build_redis_client")
    fake_schema_client = mocker.Mock(preload=mocker.Mock(), aclose=mocker.AsyncMock())
    fake_policy_client = mocker.Mock(aclose=mocker.AsyncMock())
    settings = Settings(_env_file=None)
    app, _agent = build_app(settings, schema_client=fake_schema_client, policy_client=fake_policy_client)

    with TestClient(app):
        pass

    build_redis_client.assert_not_called()
