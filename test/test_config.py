import pytest
from pydantic import ValidationError

from indykite.intent_agent.config import Settings

_DEFAULT_INTENT_PORT = 6003
_DEFAULT_HEALTH_PORT = 8001
_DEFAULT_SCHEMA_REFRESH_TTL_MINUTES = 1440.0


@pytest.fixture(autouse=True)
def _clean_settings_env(monkeypatch):
    """Strip every Settings field's env var so ambient/shell env can't leak into a test -
    each test opts back in explicitly via _set_baseline_env/monkeypatch.setenv.
    """
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)


def _set_baseline_env(monkeypatch):
    """Set every currently-required Settings env var to a valid value."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("SUBJECT_TYPE", "Customer")
    monkeypatch.setenv("IK_APP_AGENT_KEY", "test-app-agent-key")
    monkeypatch.setenv("INDYKITE_BASE_URL", "https://ikg.example.com")


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
    assert settings.cache_backend == "memory"
    assert str(settings.bind_host) == "0.0.0.0"  # noqa: S104 - asserting the default, not binding anything
    assert settings.log_level == "INFO"


def test_rejects_blank_gemini_api_key(monkeypatch):
    """A whitespace-only GEMINI_API_KEY must fail validation, not be treated as set."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_gemini_api_key_does_not_leak_in_repr(monkeypatch):
    """SecretStr must mask GEMINI_API_KEY from logs/tracebacks that repr() the Settings object."""
    _set_baseline_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert "test-key" not in repr(settings)
    assert settings.gemini_api_key.get_secret_value() == "test-key"


def test_requires_ik_app_agent_key(monkeypatch):
    """Settings must fail fast when IK_APP_AGENT_KEY is missing, not default to blank."""
    _set_baseline_env(monkeypatch)
    monkeypatch.delenv("IK_APP_AGENT_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rejects_blank_ik_app_agent_key(monkeypatch):
    """A whitespace-only IK_APP_AGENT_KEY must fail validation, not be treated as set."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("IK_APP_AGENT_KEY", "   ")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_ik_app_agent_key_does_not_leak_in_repr(monkeypatch):
    """SecretStr must mask IK_APP_AGENT_KEY from logs/tracebacks that repr() the Settings object."""
    _set_baseline_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert "test-app-agent-key" not in repr(settings)


@pytest.mark.parametrize("bad_url", ["not-a-url", "ftp://ikg.example.com"])
def test_rejects_malformed_indykite_base_url(monkeypatch, bad_url):
    """INDYKITE_BASE_URL must be a well-formed http(s) URL, not just any non-blank string."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("INDYKITE_BASE_URL", bad_url)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_requires_indykite_base_url(monkeypatch):
    """Settings must fail fast when INDYKITE_BASE_URL is missing, not default to blank."""
    _set_baseline_env(monkeypatch)
    monkeypatch.delenv("INDYKITE_BASE_URL", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rejects_blank_indykite_base_url(monkeypatch):
    """A blank/whitespace-only INDYKITE_BASE_URL must fail validation, not be treated as set."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("INDYKITE_BASE_URL", "   ")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rejects_malformed_bind_host(monkeypatch):
    """BIND_HOST must be a valid IP address, not an arbitrary hostname."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("BIND_HOST", "not-an-ip")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rejects_unknown_log_level(monkeypatch):
    """LOG_LEVEL must be one of the standard logging levels, not an arbitrary string."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("LOG_LEVEL", "TRACE")
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


def test_rejects_matching_intent_and_health_ports(monkeypatch):
    """INTENT_PORT and HEALTH_PORT must differ, since both are bound in the same process."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("INTENT_PORT", "7000")
    monkeypatch.setenv("HEALTH_PORT", "7000")
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


def test_introspect_client_secret_does_not_leak_in_repr(monkeypatch):
    """SecretStr must mask INTROSPECT_CLIENT_SECRET from logs/tracebacks that repr() Settings."""
    _set_baseline_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert "client-secret" not in repr(settings)
    assert settings.introspect_client_secret.get_secret_value() == "client-secret"


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
        pytest.param("   ", id="whitespace_only"),
    ],
)
def test_rejects_invalid_redis_url(monkeypatch, value):
    """A malformed/whitespace-only REDIS_URL, an out-of-range port, or a non-redis scheme must
    all fail validation - only a truly blank REDIS_URL disables Redis (see test_defaults).
    """
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("REDIS_URL", value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rejects_redis_backend_without_redis_url(monkeypatch):
    """CACHE_BACKEND=redis with no REDIS_URL must fail fast - there's nothing to connect to."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("CACHE_BACKEND", "redis")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_accepts_redis_backend_with_redis_url(monkeypatch):
    """CACHE_BACKEND=redis with a REDIS_URL set must be accepted."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("CACHE_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://redis.example.com:6379")
    settings = Settings(_env_file=None)
    assert settings.cache_backend == "redis"


def test_rejects_unknown_cache_backend(monkeypatch):
    """CACHE_BACKEND must be one of the two supported backends, not an arbitrary string."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("CACHE_BACKEND", "memcached")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("ttl", [0, -1])
def test_rejects_non_positive_schema_refresh_ttl(monkeypatch, ttl):
    """A zero or negative SCHEMA_REFRESH_TTL_MINUTES would busy-loop the refresh thread."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("SCHEMA_REFRESH_TTL_MINUTES", str(ttl))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "env_var",
    ["TOOL_CALL_MAX_ITERATIONS", "SCHEMA_PREVIEW_CHARS", "SCHEMA_FETCH_TIMEOUT_SECONDS", "INTROSPECT_TIMEOUT_SECONDS"],
)
@pytest.mark.parametrize("value", [0, -1])
def test_rejects_non_positive_numeric_settings(monkeypatch, env_var, value):
    """TOOL_CALL_MAX_ITERATIONS, SCHEMA_PREVIEW_CHARS, and the timeout fields must all be
    strictly positive - zero or negative would either skip the tool-call loop entirely,
    truncate every schema preview to nothing, or make an instantly-timing-out HTTP request.
    """
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv(env_var, str(value))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
