import pytest
from pydantic import ValidationError

from indykite.intent_agent.config import Settings

_DEFAULT_INTENT_PORT = 6003
_DEFAULT_HEALTH_PORT = 8001
_DEFAULT_SCHEMA_REFRESH_TTL_MINUTES = 1440.0


def _set_baseline_env(monkeypatch):
    """Set every currently-required Settings env var to a valid value."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("SUBJECT_TYPE", "Customer")


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
    assert settings.redis_url is None
    assert settings.schema_refresh_ttl_minutes == _DEFAULT_SCHEMA_REFRESH_TTL_MINUTES


def test_rejects_blank_gemini_api_key(monkeypatch):
    """A whitespace-only GEMINI_API_KEY must fail validation, not be treated as set."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("env_var", ["INTENT_PORT", "HEALTH_PORT"])
@pytest.mark.parametrize("port", [0, -1, 65536])
def test_rejects_out_of_range_ports(monkeypatch, env_var, port):
    """INTENT_PORT/HEALTH_PORT outside the valid TCP port range (1-65535) must fail validation."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv(env_var, str(port))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("env_var", ["INTROSPECT_URL", "INTROSPECT_CLIENT_ID", "INTROSPECT_CLIENT_SECRET"])
@pytest.mark.parametrize(
    "make_invalid",
    [
        pytest.param(lambda monkeypatch, env_var: monkeypatch.delenv(env_var, raising=False), id="missing"),
        pytest.param(lambda monkeypatch, env_var: monkeypatch.setenv(env_var, "   "), id="blank"),
    ],
)
def test_rejects_missing_or_blank_introspect_config(monkeypatch, env_var, make_invalid):
    """Settings must fail fast when introspection config is missing or whitespace-only,
    not default to/treat it as blank.
    """
    _set_baseline_env(monkeypatch)
    make_invalid(monkeypatch, env_var)
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


def test_requires_subject_type(monkeypatch):
    """Settings must fail fast when SUBJECT_TYPE is missing, not default to blank."""
    # Question: What about Subject search? ENG-9069
    _set_baseline_env(monkeypatch)
    monkeypatch.delenv("SUBJECT_TYPE", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rejects_blank_subject_type(monkeypatch):
    """A whitespace-only SUBJECT_TYPE must fail validation, not be treated as set."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("SUBJECT_TYPE", "   ")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("not-a-real-url", id="malformed_url"),
        pytest.param("redis://host:99999", id="port_too_large"),
        pytest.param("http://host:6379", id="wrong_scheme"),
    ],
)
def test_rejects_invalid_redis_url(monkeypatch, value):
    """A malformed REDIS_URL, an out-of-range port, or a non-redis scheme must all fail validation."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("REDIS_URL", value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("ttl", [0, -1])
def test_rejects_non_positive_schema_refresh_ttl(monkeypatch, ttl):
    """A zero or negative SCHEMA_REFRESH_TTL_MINUTES would busy-loop the refresh thread."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("SCHEMA_REFRESH_TTL_MINUTES", str(ttl))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
