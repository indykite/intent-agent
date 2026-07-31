import pytest
from pydantic import ValidationError

from indykite.intent_agent.config import Settings

_DEFAULT_INTENT_PORT = 6003
_DEFAULT_HEALTH_PORT = 8001


def test_requires_gemini_api_key(monkeypatch):
    """Settings must fail fast when GEMINI_API_KEY is missing, not default to empty."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_defaults(monkeypatch):
    """Every optional Settings field must fall back to its documented default value."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    settings = Settings(_env_file=None)
    assert settings.intent_port == _DEFAULT_INTENT_PORT
    assert settings.health_port == _DEFAULT_HEALTH_PORT
    assert settings.intent_agent_name == "intent_agent"
    assert settings.gemini_model == "gemini-2.5-flash"


def test_rejects_blank_gemini_api_key(monkeypatch):
    """A whitespace-only GEMINI_API_KEY must fail validation, not be treated as set."""
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_rejects_out_of_range_intent_port(monkeypatch, port):
    """INTENT_PORT outside the valid TCP port range (1-65535) must fail validation."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTENT_PORT", str(port))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_rejects_out_of_range_health_port(monkeypatch, port):
    """HEALTH_PORT outside the valid TCP port range (1-65535) must fail validation."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("HEALTH_PORT", str(port))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
