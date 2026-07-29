"""Runtime configuration for the Intent Agent, loaded from environment / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Startup config. Fails fast (pydantic ValidationError) if fields are missing."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    intent_port: int = 6003
    health_port: int = 8001
    bind_host: str = "0.0.0.0"  # noqa: S104 - reachable from outside container
    advertised_host: str = "intent"
    intent_agent_name: str = "intent_agent"
    log_level: str = "INFO"

    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    indykite_base_url: str = ""
    ik_app_agent_key: str = ""
