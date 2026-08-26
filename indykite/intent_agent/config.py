"""Runtime configuration for the Intent Agent, loaded from environment / .env."""

from ipaddress import IPv4Address
from typing import Annotated, Literal

from pydantic import (
    Field,
    HttpUrl,
    IPvAnyAddress,
    PositiveFloat,
    PositiveInt,
    RedisDsn,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

_Port = Annotated[int, Field(ge=1, le=65535)]
_LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


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
    tool_call_max_iterations: PositiveInt = 5
    schema_preview_chars: PositiveInt = 12000

    indykite_base_url: HttpUrl
    ik_app_agent_key: SecretStr = Field(min_length=1)
    schema_fetch_timeout_seconds: PositiveFloat = 30.0
    schema_refresh_ttl_minutes: PositiveFloat = 5.0

    cache_backend: Literal["memory", "redis"] = "memory"
    redis_url: RedisDsn | None = None

    introspect_url: HttpUrl
    introspect_client_id: str = Field(min_length=1)
    introspect_client_secret: SecretStr = Field(min_length=1)
    introspect_timeout_seconds: PositiveFloat = 5.0

    subject_type: str = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_matching_ports(self) -> "Settings":
        """Reject intent_port and health_port being the same (would collide on bind)."""
        if self.intent_port == self.health_port:
            message = "INTENT_PORT and HEALTH_PORT must be different"
            raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _require_redis_url_for_redis_backend(self) -> "Settings":
        """Reject CACHE_BACKEND=redis with no REDIS_URL - there'd be nothing to connect to."""
        if self.cache_backend == "redis" and self.redis_url is None:
            message = "REDIS_URL is required when CACHE_BACKEND is 'redis'"
            raise ValueError(message)
        return self
