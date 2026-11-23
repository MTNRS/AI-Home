"""Environment-backed application settings."""

from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @model_validator(mode="after")
    def validate_deployment_settings(self) -> "Settings":
        if self.session_lifetime_seconds < 60:
            raise ValueError("Session lifetime must be at least 60 seconds.")
        if self.environment == "production" and not self.database_url.startswith(
            ("postgresql://", "postgresql+psycopg://")
        ):
            raise ValueError("Production deployments must use PostgreSQL.")
        return self
