from indykite.intent_agent.audit.async_sender import AsyncSender
from indykite.intent_agent.audit.file import FileSender
from indykite.intent_agent.audit.http import HttpSender
from indykite.intent_agent.audit.sender import build_sender
from indykite.intent_agent.config import Settings

_BUFFER_SIZE = 16


def test_file_delivery_builds_a_file_sender(monkeypatch, tmp_path, baseline_env):
    """AUDIT_DELIVERY=file wires the file sender into the configured directory."""
    baseline_env(tmp_path / "audit")

    sender = build_sender(Settings(_env_file=None))

    assert isinstance(sender, FileSender)
    assert sender.current_path is not None
    assert sender.current_path.parent == tmp_path / "audit"


def test_webhook_delivery_builds_an_http_sender(monkeypatch, baseline_env):
    """AUDIT_DELIVERY=webhook wires the HTTP sender."""
    baseline_env()
    monkeypatch.setenv("AUDIT_DELIVERY", "webhook")
    monkeypatch.setenv("AUDIT_HTTP_URL", "https://audit.example.com/records")
    monkeypatch.setenv("AUDIT_HTTP_METHOD", "post")
    monkeypatch.setenv("AUDIT_HTTP_AUTH_TYPE", "api-key")
    monkeypatch.setenv("AUDIT_HTTP_API_KEY", "test-audit-key")

    sender = build_sender(Settings(_env_file=None))

    assert isinstance(sender, HttpSender)


def test_async_wraps_the_delivery(monkeypatch, tmp_path, baseline_env):
    """AUDIT_ASYNC=true puts the buffered queue in front of whichever delivery is configured."""
    baseline_env(tmp_path / "audit")
    monkeypatch.setenv("AUDIT_ASYNC", "true")
    monkeypatch.setenv("AUDIT_ASYNC_BUFFER_SIZE", str(_BUFFER_SIZE))

    sender = build_sender(Settings(_env_file=None))

    assert isinstance(sender, AsyncSender)
    assert sender.buffer_size == _BUFFER_SIZE
