"""Environment-backed application settings."""

from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_home.urls import normalize_provider_url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_HOME_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    database_url: str = "sqlite+pysqlite:///./ai_home.db"
    session_lifetime_seconds: int = 86_400
    chat_provider_url: str | None = None
    chat_provider_model: str | None = None
    chat_provider_api_key: SecretStr = SecretStr("")
    chat_provider_timeout_seconds: int = 180

    @model_validator(mode="after")
    def validate_deployment_settings(self) -> "Settings":
        if self.session_lifetime_seconds < 60:
            raise ValueError("Session lifetime must be at least 60 seconds.")
        if self.environment == "production" and not self.database_url.startswith(
            ("postgresql://", "postgresql+psycopg://")
        ):
            raise ValueError("Production deployments must use PostgreSQL.")
        if not 1 <= self.chat_provider_timeout_seconds <= 600:
            raise ValueError("Chat provider timeout must be between 1 and 600 seconds.")
        if self.chat_provider_url is not None:
            self.chat_provider_url = normalize_provider_url(self.chat_provider_url)
            if not self.chat_provider_model or not self.chat_provider_model.strip():
                raise ValueError("A chat provider model is required when the provider is set.")
        return self
