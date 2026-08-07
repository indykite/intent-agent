import pytest
from pydantic import ValidationError

from indykite.intent_agent.config import Settings

_DEFAULT_INTENT_PORT = 6003
_DEFAULT_HEALTH_PORT = 8001


def _set_baseline_env(monkeypatch):
    """Set every currently-required Settings env var to a valid value."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")


def test_requires_gemini_api_key(monkeypatch):
    """Settings must fail fast when GEMINI_API_KEY is missing, not default to empty."""
    _set_baseline_env(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_defaults(monkeypatch):
    """Every optional Settings field must fall back to its documented default value."""
    _set_baseline_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert settings.intent_port == _DEFAULT_INTENT_PORT
    assert settings.health_port == _DEFAULT_HEALTH_PORT
    assert settings.intent_agent_name == "intent_agent"
    assert settings.gemini_model == "gemini-2.5-flash"


def test_rejects_blank_gemini_api_key(monkeypatch):
    """A whitespace-only GEMINI_API_KEY must fail validation, not be treated as set."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_rejects_out_of_range_intent_port(monkeypatch, port):
    """INTENT_PORT outside the valid TCP port range (1-65535) must fail validation."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("INTENT_PORT", str(port))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_rejects_out_of_range_health_port(monkeypatch, port):
    """HEALTH_PORT outside the valid TCP port range (1-65535) must fail validation."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("HEALTH_PORT", str(port))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("env_var", ["INTROSPECT_URL", "INTROSPECT_CLIENT_ID", "INTROSPECT_CLIENT_SECRET"])
def test_requires_introspect_config(monkeypatch, env_var):
    """Settings must fail fast when introspection config is missing, not default to blank."""
    _set_baseline_env(monkeypatch)
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("env_var", ["INTROSPECT_URL", "INTROSPECT_CLIENT_ID", "INTROSPECT_CLIENT_SECRET"])
def test_rejects_blank_introspect_config(monkeypatch, env_var):
    """A whitespace-only introspection config value must fail validation, not be treated as set."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv(env_var, "   ")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("bad_url", ["not-a-url", "ftp://idp.example.com/introspect"])
def test_rejects_malformed_introspect_url(monkeypatch, bad_url):
    """INTROSPECT_URL must be a well-formed http(s) URL, not just any non-blank string."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("INTROSPECT_URL", bad_url)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_introspect_url_stringifies_without_mangling_path(monkeypatch):
    """INTROSPECT_URL must round-trip to str() unchanged when it already has a path."""
    _set_baseline_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert str(settings.introspect_url) == "https://idp.example.com/introspect"
