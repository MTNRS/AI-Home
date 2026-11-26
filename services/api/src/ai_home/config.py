"""Environment-backed application settings."""

from typing import Literal
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_home.urls import normalize_provider_url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_HOME_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    environment: Literal["development", "test", "production"] = "development"
    database_url: str = "sqlite+pysqlite:///./ai_home.db"
    session_lifetime_seconds: int = 86_400
    chat_backend: Literal["foundation", "openai-compatible"] = "foundation"
    foundation_checkpoint: Path | None = None
    foundation_device: Literal["auto", "cpu", "cuda"] = "auto"
    foundation_max_new_tokens: int = Field(default=128, ge=1, le=4096)
    foundation_memory_budget_bytes: int = Field(default=2_000_000_000, ge=1)
    chat_provider_url: str | None = None
    chat_provider_model: str | None = None
    chat_provider_api_key: SecretStr = SecretStr("")
    chat_provider_timeout_seconds: int = 180
    training_review_token: SecretStr = SecretStr("")
    training_reviewer_id: str = Field(
        default="configured-reviewer",
        min_length=1,
        max_length=80,
    )
    supabase_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_URL", "AI_HOME_SUPABASE_URL"),
    )
    supabase_publishable_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices(
            "SUPABASE_PUBLISHABLE_KEY",
            "AI_HOME_SUPABASE_PUBLISHABLE_KEY",
        ),
    )
    supabase_secret_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices(
            "SUPABASE_SECRET_KEY",
            "AI_HOME_SUPABASE_SECRET_KEY",
        ),
    )
    supabase_jwks_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_JWKS_URL", "AI_HOME_SUPABASE_JWKS_URL"),
    )

    @property
    def supabase_auth_enabled(self) -> bool:
        return self.supabase_url is not None

    @model_validator(mode="after")
    def validate_deployment_settings(self) -> "Settings":
        if self.session_lifetime_seconds < 60:
            raise ValueError("Session lifetime must be at least 60 seconds.")
        if self.environment == "production" and not self.database_url.startswith(
            ("postgresql://", "postgresql+psycopg://")
        ):
            raise ValueError("Production deployments must use PostgreSQL.")
        if self.environment == "production" and not self.supabase_auth_enabled:
            raise ValueError("Production deployments must configure Supabase Auth.")
        if (
            self.environment == "production"
            and not self.supabase_secret_key.get_secret_value()
        ):
            raise ValueError(
                "Production deployments must configure SUPABASE_SECRET_KEY."
            )
        if not 1 <= self.chat_provider_timeout_seconds <= 600:
            raise ValueError("Chat provider timeout must be between 1 and 600 seconds.")
        review_token = self.training_review_token.get_secret_value()
        if review_token and (len(review_token) < 32 or not review_token.isascii()):
            raise ValueError(
                "Training review token must be at least 32 ASCII characters."
            )
        if review_token and not self.training_reviewer_id.strip():
            raise ValueError("A training reviewer identifier is required.")
        supabase_publishable_key = self.supabase_publishable_key.get_secret_value()
        if self.supabase_url is None:
            if supabase_publishable_key or self.supabase_jwks_url is not None:
                raise ValueError(
                    "SUPABASE_URL is required when Supabase Auth is configured."
                )
        else:
            project = urlsplit(self.supabase_url)
            if (
                project.scheme != "https"
                or not project.hostname
                or project.username is not None
                or project.password is not None
                or project.query
                or project.fragment
                or project.path
            ):
                raise ValueError("SUPABASE_URL must be an HTTPS project URL.")
            if not supabase_publishable_key:
                raise ValueError(
                    "SUPABASE_PUBLISHABLE_KEY is required for Supabase Auth."
                )
            expected_jwks_url = (
                f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
            )
            if self.supabase_jwks_url is None:
                self.supabase_jwks_url = expected_jwks_url
            elif self.supabase_jwks_url != expected_jwks_url:
                raise ValueError(
                    "SUPABASE_JWKS_URL must match the configured Supabase project."
                )
        if self.chat_provider_url is not None:
            self.chat_provider_url = normalize_provider_url(self.chat_provider_url)
            if not self.chat_provider_model or not self.chat_provider_model.strip():
                raise ValueError("A chat provider model is required when the provider is set.")
        return self

