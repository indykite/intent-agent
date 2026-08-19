"""Runtime configuration for the Intent Agent, loaded from environment / .env."""

from pydantic import Field, HttpUrl, RedisDsn, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Startup config. Fails fast (pydantic ValidationError) if fields are missing."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    intent_port: int = Field(default=6003, ge=1, le=65535)
    health_port: int = Field(default=8001, ge=1, le=65535)
    # skipcq: BAN-B104  # noqa: ERA001 - looks like code, is a suppression directive
    bind_host: str = "0.0.0.0"  # noqa: S104 - reachable from outside container
    advertised_host: str = "intent"
    intent_agent_name: str = "intent_agent"
    log_level: str = "INFO"

    gemini_api_key: str = Field(min_length=1)
    gemini_model: str = "gemini-2.5-flash"
    tool_call_max_iterations: int = 5
    schema_preview_chars: int = 12000

    indykite_base_url: str = ""
    ik_app_agent_key: str = ""
    schema_fetch_timeout_seconds: float = 30.0
    schema_refresh_ttl_minutes: float = Field(default=1440.0, gt=0)

    redis_url: RedisDsn | None = None

    introspect_url: HttpUrl
    introspect_client_id: str = Field(min_length=1)
    introspect_client_secret: str = Field(min_length=1)
    introspect_timeout_seconds: float = 5.0

    subject_type: str = Field(min_length=1)

    @field_validator("gemini_api_key")
    @classmethod
    def _reject_blank_gemini_api_key(cls, value: str) -> str:
        """Reject a whitespace-only key (``min_length`` alone would accept one)."""
        if not value.strip():
            message = "GEMINI_API_KEY must not be blank"
            raise ValueError(message)
        return value

    @field_validator("introspect_client_id", "introspect_client_secret", "subject_type")
    @classmethod
    def _reject_blank_introspect_config(cls, value: str, info: ValidationInfo) -> str:
        """Reject a whitespace-only value (``min_length`` alone would accept one)."""
        if not value.strip():
            message = f"{info.field_name} must not be blank"
            raise ValueError(message)
        return value

    @model_validator(mode="after")
    def _reject_matching_ports(self) -> "Settings":
        """Reject intent_port and health_port being the same (would collide on bind)."""
        if self.intent_port == self.health_port:
            message = "INTENT_PORT and HEALTH_PORT must be different"
            raise ValueError(message)
        return self
