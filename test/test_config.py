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
# PRD: a request must be answered within 10s end to end. REQUEST_TIMEOUT_SECONDS enforces it;
# the per-call default is sized so the common path (skill activation, answer, one retry
# answer) fits inside it, with extra tool-call rounds still cut off by the request deadline.
_PRD_REQUEST_BUDGET_SECONDS = 10.0
_LLM_CALLS_ON_COMMON_RETRIED_PATH = 3


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
    monkeypatch.setenv("IK_APP_AGENT_KEY", "test-app-agent-key")
    monkeypatch.setenv("INDYKITE_BASE_URL", "https://ikg.example.com")
    # AUDIT_DELIVERY is required (TDD 4.9: every request produces a record); Settings never
    # creates the directory, so a placeholder path is fine here.
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
    [
        "TOOL_CALL_MAX_ITERATIONS",
        "SCHEMA_PREVIEW_CHARS",
        "SCHEMA_FETCH_TIMEOUT_SECONDS",
        "INTROSPECT_TIMEOUT_SECONDS",
        "LLM_TIMEOUT_SECONDS",
        "REQUEST_TIMEOUT_SECONDS",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_rejects_non_positive_numeric_settings(monkeypatch, env_var, value):
    """TOOL_CALL_MAX_ITERATIONS, SCHEMA_PREVIEW_CHARS, and the timeout fields must all be
    strictly positive - zero or negative would either skip the tool-call loop entirely,
    truncate every schema preview to nothing, or make an instantly-timing-out call.
    """
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv(env_var, str(value))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rejects_negative_validation_max_retries(monkeypatch):
    """VALIDATION_MAX_RETRIES is a count of extra attempts - negative is meaningless."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("VALIDATION_MAX_RETRIES", "-1")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_accepts_zero_validation_max_retries(monkeypatch):
    """Unlike the timeout fields, zero is a legitimate value here: it disables the retry."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("VALIDATION_MAX_RETRIES", "0")
    assert Settings(_env_file=None).validation_max_retries == 0


def test_default_timeouts_match_prd_request_budget(monkeypatch):
    """REQUEST_TIMEOUT_SECONDS defaults to the PRD's 10s end-to-end limit, and the per-call
    default is sized so the common retried path (activation + answer + one retry answer, each
    at the per-call limit) fits inside it - Gemini's own 10s client default would not. Guards
    against bumping either default without redoing that arithmetic.

    The schema/policy fetch defaults (ENG-9671: both moved inside this same deadline) each fit
    the same per-field rule the reject validator enforces, but their sum is not also claimed to
    fit alongside the LLM path above - the LLM path's own worst case already leaves it almost no
    room on its own, the same way it did before either fetch moved inside this deadline; the
    request-level asyncio.timeout, not this arithmetic, is what actually bounds a request that
    hits several of these ceilings at once.
    """
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
    """A per-call limit above the whole-request deadline could never fire first - dead config,
    for the LLM call and, since ENG-9671 moved both inside the same deadline, the schema and
    policy fetch too.
    """
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv(env_var, value)
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "10")
    with pytest.raises(ValidationError, match=f"{env_var} must not exceed REQUEST_TIMEOUT_SECONDS"):
        Settings(_env_file=None)


def test_rejects_a_combined_fetch_worst_case_even_when_each_field_alone_fits(monkeypatch):
    """A schema timeout plus policy_fetch_timeout_seconds x policy_fetch_retry_attempts can
    exceed request_timeout_seconds even when each field individually stays under it (review) -
    e.g. a 9s schema fetch plus two 9s policy attempts is 27s, nowhere near a 10s request
    deadline, though 9 <= 10 passes the single-field checks above on its own.
    """
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("SCHEMA_FETCH_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("POLICY_FETCH_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("POLICY_FETCH_RETRY_ATTEMPTS", "2")
    with pytest.raises(ValidationError, match="POLICY_FETCH_RETRY_ATTEMPTS must not exceed REQUEST_TIMEOUT_SECONDS"):
        Settings(_env_file=None)


_DEFAULT_AUDIT_ASYNC_BUFFER_SIZE = 512
_DEFAULT_AUDIT_ROTATION_INTERVAL_SECONDS = 24 * 60 * 60.0
_DEFAULT_AUDIT_ROTATION_MAX_BYTES = 100 * 1024 * 1024
_DEFAULT_AUDIT_HTTP_TIMEOUT_SECONDS = 5.0


def test_requires_audit_delivery(monkeypatch):
    """Unlike the IAG (audit section absent = auditing off), AUDIT_DELIVERY has no default: the
    TDD requires every request to produce an audit record, so a deployment can't silently
    run without one configured.
    """
    _set_baseline_env(monkeypatch)
    monkeypatch.delenv("AUDIT_DELIVERY", raising=False)
    with pytest.raises(ValidationError, match="audit_delivery"):
        Settings(_env_file=None)


def test_rejects_unknown_audit_delivery(monkeypatch):
    """AUDIT_DELIVERY is `file` or `webhook`, the IAG's two delivery types - nothing else."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv("AUDIT_DELIVERY", "stdout")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_audit_defaults_match_the_iag(monkeypatch):
    """Every optional audit setting defaults to the IAG's documented default."""
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
    """AUDIT_DELIVERY=file without path/format/rotation fails at startup (IAG FileConfig rules)."""
    _set_baseline_env(monkeypatch)
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(ValidationError, match="audit: "):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    ("env_var", "value"),
    [("AUDIT_FILE_FORMAT", "yaml"), ("AUDIT_FILE_ROTATION_STRATEGY", "daily")],
)
def test_rejects_unknown_file_format_or_rotation(monkeypatch, env_var, value):
    """Formats are csv/json/txt and rotation is time/size/size-and-time, as documented for the IAG."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv(env_var, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "env_var",
    ["AUDIT_ASYNC_BUFFER_SIZE", "AUDIT_FILE_ROTATION_INTERVAL_SECONDS", "AUDIT_FILE_ROTATION_MAX_BYTES"],
)
@pytest.mark.parametrize("value", [0, -1])
def test_rejects_non_positive_audit_limits(monkeypatch, env_var, value):
    """A zero/negative queue size, rotation interval or size limit would rotate or drop on every record."""
    _set_baseline_env(monkeypatch)
    monkeypatch.setenv(env_var, str(value))
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
    """AUDIT_DELIVERY=webhook without url/method/auth fails at startup (IAG HTTPConfig rules)."""
    _set_baseline_env(monkeypatch)
    _set_webhook_env(monkeypatch)
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(ValidationError, match=expected):
        Settings(_env_file=None)


def test_webhook_method_is_case_insensitive(monkeypatch):
    """`post`/`put` in any case are accepted and normalised, like the IAG's strings.ToUpper."""
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


def test_mtls_requires_certificate_and_key(monkeypatch):
    """The IAG's rule, verbatim in spirit: mTLS needs both files, named at startup."""
    _set_baseline_env(monkeypatch)
    _set_webhook_env(monkeypatch, auth_type="mTLS")
    with pytest.raises(ValidationError, match="mTLS requires"):
        Settings(_env_file=None)


def test_mtls_files_must_exist(monkeypatch, tmp_path):
    """A certificate path that doesn't exist is a startup error, not a first-record surprise."""
    _set_baseline_env(monkeypatch)
    _set_webhook_env(monkeypatch, auth_type="mTLS")
    monkeypatch.setenv("AUDIT_HTTP_MTLS_CERTIFICATE_FILE_PATH", str(tmp_path / "missing-cert.pem"))
    monkeypatch.setenv("AUDIT_HTTP_MTLS_PRIVATE_KEY_PATH", str(tmp_path / "missing-key.pem"))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_api_key_auth_requires_a_key(monkeypatch):
    """The IAG's rule: api-key auth without a key is a config error."""
    _set_baseline_env(monkeypatch)
    _set_webhook_env(monkeypatch, auth_type="api-key")
    with pytest.raises(ValidationError, match="api-key auth requires AUDIT_HTTP_API_KEY"):
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
    """Auth is mTLS, no-auth or api-key - the IAG's three."""
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
