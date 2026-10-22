import pytest
from pydantic import ValidationError

from indykite.intent_agent.config import Settings

_DEFAULT_INTENT_PORT = 6003
_DEFAULT_HEALTH_PORT = 8001
_DEFAULT_SCHEMA_REFRESH_TTL_MINUTES = 5.0
_DEFAULT_POLICY_REFRESH_TTL_MINUTES = 5.0
_DEFAULT_POLICY_FETCH_RETRY_ATTEMPTS = 2
_DEFAULT_LLM_TIMEOUT_SECONDS = 3.0
_DEFAULT_SCHEMA_FETCH_TIMEOUT_SECONDS = 3.0
_DEFAULT_POLICY_FETCH_TIMEOUT_SECONDS = 2.0
_DEFAULT_VALIDATION_MAX_RETRIES = 1
# The common path (skill activation, answer, one retry answer) must fit the 10s request budget.
_PRD_REQUEST_BUDGET_SECONDS = 10.0
_LLM_CALLS_ON_COMMON_RETRIED_PATH = 3


@pytest.fixture(autouse=True)
def _clean_settings_env(monkeypatch):
    """Strip every Settings field's env var so the shell environment can't leak into a test."""
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)


def _set_baseline_env(monkeypatch):
    """Set every currently-required Settings env var to a valid value."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("INTROSPECT_URL", "https://idp.example.com/introspect")
    monkeypatch.setenv("INTROSPECT_CLIENT_ID", "client-id")
    monkeypatch.setenv("INTROSPECT_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("IK_APP_AGENT_KEY", "test-app-agent-key")
    monkeypatch.setenv("INDYKITE_BASE_URL", "https://ikg.example.com")
    # Settings never creates the directory, so a placeholder path is fine.
    monkeypatch.setenv("AUDIT_DELIVERY", "file")
    monkeypatch.setenv("AUDIT_FILE_STORAGE_PATH", "./.private/audit-never-created")
    monkeypatch.setenv("AUDIT_FILE_FORMAT", "json")
    monkeypatch.setenv("AUDIT_FILE_ROTATION_STRATEGY", "time")


def _set_webhook_env(monkeypatch, auth_type="no-auth"):
    """Switch the baseline to webhook delivery with the given auth type."""
    monkeypatch.setenv("AUDIT_DELIVERY", "webhook")
    monkeypatch.setenv("AUDIT_HTTP_URL", "https://audit.example.com/records")
    monkeypatch.setenv("AUDIT_HTTP_METHOD", "POST")
    monkeypatch.setenv("AUDIT_HTTP_AUTH_TYPE", auth_type)


@pytest.mark.parametrize("env_var", ["GEMINI_API_KEY", "IK_APP_AGENT_KEY", "INDYKITE_BASE_URL"])
def test_requires_setting(monkeypatch, env_var):
    """Settings fails fast when a required setting is missing, instead of defaulting to blank."""
    _set_baseline_env(monkeypatch)
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_defaults(monkeypatch):
    """Every optional Settings field must fall back to its documented default value."""
    _set_baseline_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert settings.intent_port == _DEFAULT_INTENT_PORT
    assert settings.health_port == _DEFAULT_HEALTH_PORT
    assert settings.intent_agent_name == "intent_agent"
    assert settings.gemini_model == "gemini-3.5-flash-lite"
    assert settings.redis_url is None
    assert settings.schema_refresh_ttl_minutes == _DEFAULT_SCHEMA_REFRESH_TTL_MINUTES
    assert settings.policy_refresh_ttl_minutes == _DEFAULT_POLICY_REFRESH_TTL_MINUTES
    assert settings.policy_fetch_retry_attempts == _DEFAULT_POLICY_FETCH_RETRY_ATTEMPTS
    assert settings.cache_backend == "memory"
    assert settings.llm_timeout_seconds == _DEFAULT_LLM_TIMEOUT_SECONDS
    assert settings.schema_fetch_timeout_seconds == _DEFAULT_SCHEMA_FETCH_TIMEOUT_SECONDS
    assert settings.policy_fetch_timeout_seconds == _DEFAULT_POLICY_FETCH_TIMEOUT_SECONDS
    assert settings.request_timeout_seconds == _PRD_REQUEST_BUDGET_SECONDS
    assert settings.validation_max_retries == _DEFAULT_VALIDATION_MAX_RETRIES
    assert str(settings.bind_host) == "0.0.0.0"  # noqa: S104 - asserting the default, not binding anything
    assert settings.log_level == "INFO"
    assert settings.expected_token_audience == "intent_agent"  # noqa: S105 - an aud value, not a secret


@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        pytest.param("GEMINI_API_KEY", "   ", id="blank_gemini_api_key"),
        pytest.param("IK_APP_AGENT_KEY", "   ", id="blank_ik_app_agent_key"),
        pytest.param("INDYKITE_BASE_URL", "   ", id="blank_indykite_base_url"),
        pytest.param("BIND_HOST", "not-an-ip", id="malformed_bind_host"),
        pytest.param("LOG_LEVEL", "TRACE", id="unknown_log_level"),
        pytest.param("EXPECTED_TOKEN_AUDIENCE", "   ", id="blank_expected_token_audience"),
        pytest.param("CACHE_BACKEND", "redis", id="redis_backend_without_redis_url"),
        pytest.param("CACHE_BACKEND", "memcached", id="unknown_cache_backend"),
        pytest.param("VALIDATION_MAX_RETRIES", "-1", id="negative_validation_max_retries"),
        pytest.param("AUDIT_DELIVERY", "stdout", id="unknown_audit_delivery"),
    ],
)
def test_rejects_invalid_setting(monkeypatch, env_var, value):
    """A blank, malformed or unsupported setting fails validation at startup."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv(env_var, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_gemini_api_key_does_not_leak_in_repr(monkeypatch):
    """SecretStr must mask GEMINI_API_KEY from logs/tracebacks that repr() the Settings object."""
    _set_baseline_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert "test-key" not in repr(settings)
    assert settings.gemini_api_key.get_secret_value() == "test-key"


def test_ik_app_agent_key_does_not_leak_in_repr(monkeypatch):
    """SecretStr must mask IK_APP_AGENT_KEY from logs/tracebacks that repr() the Settings object."""
    _set_baseline_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert "test-app-agent-key" not in repr(settings)


@pytest.mark.parametrize(
    ("env_var", "bad_url"),
    [
        ("INDYKITE_BASE_URL", "not-a-url"),
        ("INDYKITE_BASE_URL", "ftp://ikg.example.com"),
        ("INTROSPECT_URL", "not-a-url"),
        ("INTROSPECT_URL", "ftp://idp.example.com/introspect"),
    ],
)
def test_rejects_malformed_url(monkeypatch, env_var, bad_url):
    """URL settings must be well-formed http(s) URLs, not just any non-blank string."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv(env_var, bad_url)
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
    """Missing or whitespace-only introspection config must fail validation."""
    _set_baseline_env(monkeypatch)
    make_invalid(monkeypatch, env_var)
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


def test_expected_token_audience_is_overridable(monkeypatch):
    """EXPECTED_TOKEN_AUDIENCE overrides the intent_agent_name-matching default."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("EXPECTED_TOKEN_AUDIENCE", "custom-resource-id")
    settings = Settings(_env_file=None)
    assert settings.expected_token_audience == "custom-resource-id"  # noqa: S105 - an aud value, not a secret


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
    """A malformed, whitespace-only, out-of-range-port or non-redis REDIS_URL must fail validation."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("REDIS_URL", value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_accepts_redis_backend_with_redis_url(monkeypatch):
    """CACHE_BACKEND=redis with a REDIS_URL set must be accepted."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("CACHE_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://redis.example.com:6379")
    settings = Settings(_env_file=None)
    assert settings.cache_backend == "redis"


@pytest.mark.parametrize("ttl", [0, -1])
def test_rejects_non_positive_schema_refresh_ttl(monkeypatch, ttl):
    """A zero or negative SCHEMA_REFRESH_TTL_MINUTES would busy-loop the refresh thread."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("SCHEMA_REFRESH_TTL_MINUTES", str(ttl))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "env_var",
    [
        "TOOL_CALL_MAX_ITERATIONS",
        "SCHEMA_PREVIEW_CHARS",
        "SCHEMA_FETCH_TIMEOUT_SECONDS",
        "INTROSPECT_TIMEOUT_SECONDS",
        "LLM_TIMEOUT_SECONDS",
        "REQUEST_TIMEOUT_SECONDS",
        "AUDIT_ASYNC_BUFFER_SIZE",
        "AUDIT_FILE_ROTATION_INTERVAL_SECONDS",
        "AUDIT_FILE_ROTATION_MAX_BYTES",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_rejects_non_positive_numeric_settings(monkeypatch, env_var, value):
    """Iteration, preview-size, timeout and audit-limit settings must be strictly positive."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv(env_var, str(value))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_accepts_zero_validation_max_retries(monkeypatch):
    """Unlike the timeout fields, zero is a legitimate value here: it disables the retry."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("VALIDATION_MAX_RETRIES", "0")
    assert Settings(_env_file=None).validation_max_retries == 0


def test_default_timeouts_match_prd_request_budget(monkeypatch):
    """Default timeouts fit the 10s request budget: three LLM calls together, each fetch on its own."""
    _set_baseline_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert settings.request_timeout_seconds <= _PRD_REQUEST_BUDGET_SECONDS
    assert settings.validation_max_retries == _DEFAULT_VALIDATION_MAX_RETRIES
    assert settings.llm_timeout_seconds * _LLM_CALLS_ON_COMMON_RETRIED_PATH <= settings.request_timeout_seconds
    assert settings.schema_fetch_timeout_seconds <= settings.request_timeout_seconds
    assert settings.policy_fetch_timeout_seconds <= settings.request_timeout_seconds


@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        ("LLM_TIMEOUT_SECONDS", "11"),
        ("SCHEMA_FETCH_TIMEOUT_SECONDS", "11"),
        ("POLICY_FETCH_TIMEOUT_SECONDS", "11"),
    ],
)
def test_rejects_a_fetch_or_llm_timeout_longer_than_request_timeout(monkeypatch, env_var, value):
    """A per-call LLM or fetch timeout above REQUEST_TIMEOUT_SECONDS is rejected as dead config."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv(env_var, value)
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "10")
    with pytest.raises(ValidationError, match=f"{env_var} must not exceed REQUEST_TIMEOUT_SECONDS"):
        Settings(_env_file=None)


def test_accepts_fetch_timeouts_whose_retry_multiplied_sum_exceeds_the_request_budget(monkeypatch):
    """Fetch timeouts are checked per field only; the request deadline bounds their combined worst case."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("SCHEMA_FETCH_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("POLICY_FETCH_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("POLICY_FETCH_RETRY_ATTEMPTS", "2")
    settings = Settings(_env_file=None)
    assert settings.schema_fetch_timeout_seconds == 9.0  # noqa: PLR2004 - the env value just set
    assert settings.policy_fetch_timeout_seconds == 9.0  # noqa: PLR2004 - the env value just set


_DEFAULT_AUDIT_ASYNC_BUFFER_SIZE = 512
_DEFAULT_AUDIT_ROTATION_INTERVAL_SECONDS = 24 * 60 * 60.0
_DEFAULT_AUDIT_ROTATION_MAX_BYTES = 100 * 1024 * 1024
_DEFAULT_AUDIT_HTTP_TIMEOUT_SECONDS = 5.0


def test_requires_audit_delivery(monkeypatch):
    """AUDIT_DELIVERY has no default, so a deployment can't silently run unaudited."""
    _set_baseline_env(monkeypatch)
    monkeypatch.delenv("AUDIT_DELIVERY", raising=False)
    with pytest.raises(ValidationError, match="audit_delivery"):
        Settings(_env_file=None)


def test_audit_defaults_match_the_iag(monkeypatch):
    """Every optional audit setting falls back to its documented default."""
    _set_baseline_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert settings.audit_async is False
    assert settings.audit_async_buffer_size == _DEFAULT_AUDIT_ASYNC_BUFFER_SIZE
    assert settings.audit_http_api_key_header == "X-API-Key"
    assert settings.audit_http_timeout_seconds == _DEFAULT_AUDIT_HTTP_TIMEOUT_SECONDS
    assert settings.audit_file_rotation_interval_seconds == _DEFAULT_AUDIT_ROTATION_INTERVAL_SECONDS
    assert settings.audit_file_rotation_max_bytes == _DEFAULT_AUDIT_ROTATION_MAX_BYTES


@pytest.mark.parametrize(
    "env_var",
    ["AUDIT_FILE_STORAGE_PATH", "AUDIT_FILE_FORMAT", "AUDIT_FILE_ROTATION_STRATEGY"],
)
def test_file_delivery_requires_its_settings(monkeypatch, env_var):
    """AUDIT_DELIVERY=file without path/format/rotation fails at startup."""
    _set_baseline_env(monkeypatch)
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(ValidationError, match="audit: "):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    ("env_var", "value"),
    [("AUDIT_FILE_FORMAT", "yaml"), ("AUDIT_FILE_ROTATION_STRATEGY", "daily")],
)
def test_rejects_unknown_file_format_or_rotation(monkeypatch, env_var, value):
    """Formats are csv/json/txt and rotation is time/size/size-and-time."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv(env_var, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    ("env_var", "expected"),
    [
        ("AUDIT_HTTP_URL", "AUDIT_HTTP_URL is required"),
        ("AUDIT_HTTP_METHOD", "must be POST or PUT"),
        ("AUDIT_HTTP_AUTH_TYPE", "AUDIT_HTTP_AUTH_TYPE is required"),
    ],
)
def test_webhook_delivery_requires_its_settings(monkeypatch, env_var, expected):
    """AUDIT_DELIVERY=webhook without url/method/auth fails at startup."""
    _set_baseline_env(monkeypatch)
    _set_webhook_env(monkeypatch)
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(ValidationError, match=expected):
        Settings(_env_file=None)


def test_webhook_method_is_case_insensitive(monkeypatch):
    """`post`/`put` in any case are accepted and normalised to upper case."""
    _set_baseline_env(monkeypatch)
    _set_webhook_env(monkeypatch)
    monkeypatch.setenv("AUDIT_HTTP_METHOD", "put")
    assert Settings(_env_file=None).audit_http_method == "PUT"


@pytest.mark.parametrize("method", ["GET", "PATCH", "DELETE"])
def test_rejects_other_webhook_methods(monkeypatch, method):
    """Only POST and PUT can carry a body the receiver is expected to store."""
    _set_baseline_env(monkeypatch)
    _set_webhook_env(monkeypatch)
    monkeypatch.setenv("AUDIT_HTTP_METHOD", method)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    ("auth_type", "message"),
    [("mTLS", "mTLS requires"), ("api-key", "api-key auth requires AUDIT_HTTP_API_KEY")],
)
def test_webhook_auth_requires_credentials(monkeypatch, auth_type, message):
    """mTLS needs a certificate and a key file; api-key auth needs a key."""
    _set_baseline_env(monkeypatch)
    _set_webhook_env(monkeypatch, auth_type=auth_type)
    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None)


def test_mtls_files_must_exist(monkeypatch, tmp_path):
    """A certificate path that doesn't exist is a startup error, not a first-record surprise."""
    _set_baseline_env(monkeypatch)
    _set_webhook_env(monkeypatch, auth_type="mTLS")
    monkeypatch.setenv("AUDIT_HTTP_MTLS_CERTIFICATE_FILE_PATH", str(tmp_path / "missing-cert.pem"))
    monkeypatch.setenv("AUDIT_HTTP_MTLS_PRIVATE_KEY_PATH", str(tmp_path / "missing-key.pem"))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_audit_api_key_does_not_leak_in_repr(monkeypatch):
    """SecretStr must mask AUDIT_HTTP_API_KEY from logs/tracebacks that repr() Settings."""
    _set_baseline_env(monkeypatch)
    _set_webhook_env(monkeypatch, auth_type="api-key")
    monkeypatch.setenv("AUDIT_HTTP_API_KEY", "audit-secret")
    settings = Settings(_env_file=None)
    assert "audit-secret" not in repr(settings)
    assert settings.audit_http_api_key.get_secret_value() == "audit-secret"


def test_rejects_unknown_audit_auth_type(monkeypatch):
    """Audit auth type must be mTLS, no-auth or api-key."""
    _set_baseline_env(monkeypatch)
    _set_webhook_env(monkeypatch, auth_type="basic")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_api_key_auth_rejects_blank_key(monkeypatch):
    """A whitespace-only AUDIT_HTTP_API_KEY is as useless as a missing one and fails at startup too."""
    _set_baseline_env(monkeypatch)
    _set_webhook_env(monkeypatch, auth_type="api-key")
    monkeypatch.setenv("AUDIT_HTTP_API_KEY", "   ")
    with pytest.raises(ValidationError, match="api-key auth requires AUDIT_HTTP_API_KEY"):
        Settings(_env_file=None)
