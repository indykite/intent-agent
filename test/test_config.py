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
