"""Runtime configuration for the Intent Agent, loaded from environment / .env."""

from ipaddress import IPv4Address
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    Field,
    FilePath,
    HttpUrl,
    IPvAnyAddress,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    RedisDsn,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

_Port = Annotated[int, Field(ge=1, le=65535)]
_LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_AuditDelivery = Literal["file", "webhook"]
_AuditHttpMethod = Literal["POST", "PUT"]
_AuditHttpAuthType = Literal["mTLS", "no-auth", "api-key"]
_AuditFileFormat = Literal["csv", "json", "txt"]
_AuditFileRotationStrategy = Literal["time", "size", "size-and-time"]

_ONE_DAY_SECONDS = 24 * 60 * 60
_100_MIB = 100 * 1024 * 1024


class Settings(BaseSettings):
    """Startup config. Fails fast (pydantic ValidationError) if fields are missing."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True, str_strip_whitespace=True)

    intent_port: _Port = 6003
    health_port: _Port = 8001
    # skipcq: BAN-B104  # noqa: ERA001 - looks like code, is a suppression directive
    bind_host: IPvAnyAddress = IPv4Address("0.0.0.0")  # noqa: S104 - reachable from outside container
    advertised_host: str = "intent"
    intent_agent_name: str = "intent_agent"
    log_level: _LogLevel = "INFO"

    gemini_api_key: SecretStr = Field(min_length=1)
    gemini_model: str = Field(min_length=1, default="gemini-3.5-flash-lite")
    # Size of IntentAgent's dedicated schema/policy-fetch thread pool (review: was a hardcoded
    # module constant). 8 is not a request-concurrency limit - it only bounds how many
    # schema/policy fetches can be in flight at once; a slow/unreachable platform ties up a
    # worker for its own timeout x retry course (see IntentAgent._run_in_fetch_executor), so
    # this is sized to absorb a burst of concurrent cold caches without spending as many threads
    # as the LLM client's own connection pool, not to cap overall request throughput.
    fetch_executor_max_workers: PositiveInt = 8
    tool_call_max_iterations: PositiveInt = 5
    schema_preview_chars: PositiveInt = 12000
    # Per-LLM-call ceiling - mirrors the other *_TIMEOUT_SECONDS fields, which each bound a
    # single outbound call. 3s is sized for the common path (skill activation, answer, one
    # validation retry) to fit the request budget below; Gemini's own 10s client default
    # would not. It is not the end-to-end guarantee: extra tool-call rounds (up to
    # TOOL_CALL_MAX_ITERATIONS per attempt) add calls, and REQUEST_TIMEOUT_SECONDS is what
    # actually caps the whole request.
    llm_timeout_seconds: PositiveFloat = 3.0
    # Whole-extraction deadline (PRD: answer within 10s end to end), spanning the schema/policy
    # fetch (ENG-9671: both moved inside this deadline, see IntentAgent.run_extraction) and
    # every LLM call, tool-call round and validation retry of one request. Introspection keeps
    # its own timeout, ahead of this one.
    request_timeout_seconds: PositiveFloat = 10.0
    # Extra attempts after the first, so the LLM is invoked at most 1 + this many times per
    # request on validation failure (TDD 4.9, open decision 1: default N=1). 0 disables retry.
    validation_max_retries: NonNegativeInt = 1

    indykite_base_url: HttpUrl
    ik_app_agent_key: SecretStr = Field(min_length=1)
    # Per-fetch ceiling, same "must fit inside request_timeout_seconds" rule as
    # llm_timeout_seconds (see _reject_fetch_timeouts_over_request_timeout) - sized to leave
    # the LLM calls a real share of the 10s budget alongside it, not to be a generous allowance
    # in its own right. Tune against the platform's actual observed latency.
    schema_fetch_timeout_seconds: PositiveFloat = 3.0
    schema_refresh_ttl_minutes: PositiveFloat = 5.0

    # Per-attempt ceiling, same rule as schema_fetch_timeout_seconds above - policy_fetch_retry_
    # attempts multiplies it, so both are sized together to leave room for the schema fetch and
    # the LLM calls in the same request_timeout_seconds budget, not just the first attempt.
    policy_fetch_timeout_seconds: PositiveFloat = 2.0
    policy_refresh_ttl_minutes: PositiveFloat = 5.0
    policy_fetch_retry_attempts: PositiveInt = 2
    policy_preview_chars: PositiveInt = 12000

    cache_backend: Literal["memory", "redis"] = "memory"
    redis_url: RedisDsn | None = None

    introspect_url: HttpUrl
    introspect_client_id: str = Field(min_length=1)
    introspect_client_secret: SecretStr = Field(min_length=1)
    introspect_timeout_seconds: PositiveFloat = 5.0

    # Audit log delivery (TDD 4.9 record, IAG-shaped delivery - see audit/). The keys mirror the
    # Agent Gateway's `audit.` config section one to one, flattened to AUDIT_* env vars. Unlike
    # the IAG, where omitting the section disables auditing, AUDIT_DELIVERY has no default:
    # the TDD requires every request to produce one audit record, so a deployment can't
    # silently run without a delivery configured.
    audit_delivery: _AuditDelivery
    # Buffered delivery through a bounded in-memory queue drained by a background worker
    # (IAG `audit.async`); records are dropped with a warning once the queue is full.
    audit_async: bool = False
    audit_async_buffer_size: PositiveInt = 512
    # Webhook delivery (IAG `audit.http.*`) - all required when AUDIT_DELIVERY is `webhook`.
    audit_http_url: HttpUrl | None = None
    audit_http_method: _AuditHttpMethod | None = None
    audit_http_auth_type: _AuditHttpAuthType | None = None
    audit_http_mtls_certificate_file_path: FilePath | None = None
    audit_http_mtls_private_key_path: FilePath | None = None
    audit_http_api_key: SecretStr | None = None
    audit_http_api_key_header: str = Field(min_length=1, default="X-API-Key")
    audit_http_timeout_seconds: PositiveFloat = 5.0
    # File delivery (IAG `audit.file.*`) - path/format/rotation required when AUDIT_DELIVERY is
    # `file`. The directory is created at startup if missing.
    audit_file_storage_path: Path | None = None
    audit_file_format: _AuditFileFormat | None = None
    audit_file_rotation_strategy: _AuditFileRotationStrategy | None = None
    audit_file_rotation_interval_seconds: PositiveFloat = float(_ONE_DAY_SECONDS)
    audit_file_rotation_max_bytes: PositiveInt = _100_MIB

    @field_validator("audit_http_method", mode="before")
    @classmethod
    def _uppercase_http_method(cls, value: object) -> object:
        """Accept `post`/`put` in any case, like the IAG does, before the Literal check."""
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _require_webhook_config(self) -> "Settings":
        """Reject AUDIT_DELIVERY=webhook without the HTTP settings the sender needs, at startup
        rather than on the first record it fails to deliver (same rules as the IAG's HTTPConfig).
        """
        if self.audit_delivery != "webhook":
            return self
        if self.audit_http_url is None:
            message = "audit: AUDIT_HTTP_URL is required when AUDIT_DELIVERY is 'webhook'"
            raise ValueError(message)
        if self.audit_http_method is None:
            message = "audit: AUDIT_HTTP_METHOD must be POST or PUT when AUDIT_DELIVERY is 'webhook'"
            raise ValueError(message)
        if self.audit_http_auth_type is None:
            message = "audit: AUDIT_HTTP_AUTH_TYPE is required when AUDIT_DELIVERY is 'webhook'"
            raise ValueError(message)
        if self.audit_http_auth_type == "mTLS" and (
            self.audit_http_mtls_certificate_file_path is None or self.audit_http_mtls_private_key_path is None
        ):
            message = "audit: mTLS requires AUDIT_HTTP_MTLS_CERTIFICATE_FILE_PATH and AUDIT_HTTP_MTLS_PRIVATE_KEY_PATH"
            raise ValueError(message)
        # SecretStr isn't stripped by str_strip_whitespace, so a whitespace-only key has to be
        # caught here rather than surfacing later as a bare ValueError from the sender.
        if self.audit_http_auth_type == "api-key" and (
            self.audit_http_api_key is None or not self.audit_http_api_key.get_secret_value().strip()
        ):
            message = "audit: api-key auth requires AUDIT_HTTP_API_KEY"
            raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _require_file_config(self) -> "Settings":
        """Reject AUDIT_DELIVERY=file without a storage path, format and rotation strategy (same
        rules as the IAG's FileConfig).
        """
        if self.audit_delivery != "file":
            return self
        if self.audit_file_storage_path is None:
            message = "audit: AUDIT_FILE_STORAGE_PATH is required when AUDIT_DELIVERY is 'file'"
            raise ValueError(message)
        if self.audit_file_format is None:
            message = "audit: AUDIT_FILE_FORMAT must be one of 'csv', 'json', 'txt' when AUDIT_DELIVERY is 'file'"
            raise ValueError(message)
        if self.audit_file_rotation_strategy is None:
            message = (
                "audit: AUDIT_FILE_ROTATION_STRATEGY is required and must be one of 'time', 'size', 'size-and-time' "
                "when AUDIT_DELIVERY is 'file'"
            )
            raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _reject_matching_ports(self) -> "Settings":
        """Reject intent_port and health_port being the same (would collide on bind)."""
        if self.intent_port == self.health_port:
            message = "INTENT_PORT and HEALTH_PORT must be different"
            raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _reject_llm_timeout_over_request_timeout(self) -> "Settings":
        """Reject a per-call timeout longer than the whole-request deadline - the per-call
        limit could then never fire before the request one, so it would be dead config.
        """
        if self.llm_timeout_seconds > self.request_timeout_seconds:
            message = "LLM_TIMEOUT_SECONDS must not exceed REQUEST_TIMEOUT_SECONDS"
            raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _reject_fetch_timeouts_over_request_timeout(self) -> "Settings":
        """Reject a schema/policy fetch timeout longer than the whole-request deadline - same
        dead-config reasoning as _reject_llm_timeout_over_request_timeout, now that both fetches
        run inside request_timeout_seconds too (ENG-9671, review).

        Deliberately per-field only, not a combined/retry-multiplied budget (review: a prior
        version of this validator also rejected schema_fetch_timeout_seconds +
        policy_fetch_timeout_seconds x policy_fetch_retry_attempts exceeding
        request_timeout_seconds - reverted, since that check was inconsistent with this exact
        sibling validator and every other budget knob here: LLM_TIMEOUT_SECONDS is checked the
        same single-field way despite TOOL_CALL_MAX_ITERATIONS/VALIDATION_MAX_RETRIES letting
        several LLM calls multiply its own worst case well past request_timeout_seconds too, and
        nothing rejects that combination either. The one property every one of these fields
        actually needs is "could this single call ever fire before the whole request already
        timed out" - asyncio.timeout(request_timeout_seconds) is what truly enforces the total
        budget, cutting off whichever phase is in flight and reporting it correctly (see
        run_extraction's phase tracking), not a requirement that every possible combination of
        retries/iterations mathematically sums to fit underneath it.
        """
        if self.schema_fetch_timeout_seconds > self.request_timeout_seconds:
            message = "SCHEMA_FETCH_TIMEOUT_SECONDS must not exceed REQUEST_TIMEOUT_SECONDS"
            raise ValueError(message)
        if self.policy_fetch_timeout_seconds > self.request_timeout_seconds:
            message = "POLICY_FETCH_TIMEOUT_SECONDS must not exceed REQUEST_TIMEOUT_SECONDS"
            raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _require_redis_url_for_redis_backend(self) -> "Settings":
        """Reject CACHE_BACKEND=redis with no REDIS_URL - there'd be nothing to connect to."""
        if self.cache_backend == "redis" and self.redis_url is None:
            message = "REDIS_URL is required when CACHE_BACKEND is 'redis'"
            raise ValueError(message)
        return self
