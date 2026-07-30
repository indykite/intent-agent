"""Runtime configuration for the Intent Agent, loaded from environment / .env."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Startup config. Fails fast (pydantic ValidationError) if fields are missing."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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

    @field_validator("gemini_api_key")
    @classmethod
    def _reject_blank_gemini_api_key(cls, value: str) -> str:
        """Reject a whitespace-only key (``min_length`` alone would accept one)."""
        if not value.strip():
            message = "GEMINI_API_KEY must not be blank"
            raise ValueError(message)
        return value
