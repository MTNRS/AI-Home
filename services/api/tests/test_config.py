import pytest
from pydantic import ValidationError

from ai_home.config import Settings


def test_production_settings_require_postgresql():
    with pytest.raises(ValidationError):
        Settings(environment="production", database_url="sqlite+pysqlite:///./local.db")


def test_production_settings_accept_postgresql():
    settings = Settings(
        environment="production",
        database_url="postgresql+psycopg://user:password@localhost/ai_home",
        supabase_url="https://project.supabase.co",
        supabase_publishable_key="sb_publishable_test",
        supabase_secret_key="sb_secret_test",
    )
    assert settings.environment == "production"


def test_production_settings_require_supabase_auth_and_server_secret():
    with pytest.raises(ValidationError):
        Settings(
            environment="production",
            database_url="postgresql+psycopg://user:password@localhost/ai_home",
        )

    with pytest.raises(ValidationError):
        Settings(
            environment="production",
            database_url="postgresql+psycopg://user:password@localhost/ai_home",
            supabase_url="https://project.supabase.co",
            supabase_publishable_key="sb_publishable_test",
        )


def test_session_lifetime_has_a_safe_minimum():
    with pytest.raises(ValidationError):
        Settings(environment="test", session_lifetime_seconds=30)


def test_supabase_auth_settings_derive_the_project_jwks_url():
    settings = Settings(
        environment="test",
        supabase_url="https://project.supabase.co",
        supabase_publishable_key="sb_publishable_test",
    )

    assert settings.supabase_auth_enabled
    assert settings.supabase_jwks_url == (
        "https://project.supabase.co/auth/v1/.well-known/jwks.json"
    )


def test_supabase_settings_load_from_documented_environment_names(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_test")

    settings = Settings(_env_file=None, environment="test")

    assert settings.supabase_url == "https://project.supabase.co"
    assert settings.supabase_publishable_key.get_secret_value() == "sb_publishable_test"
    assert settings.supabase_secret_key.get_secret_value() == "sb_secret_test"


def test_supabase_auth_requires_publishable_key_and_project_scoped_jwks():
    with pytest.raises(ValidationError):
        Settings(environment="test", supabase_url="https://project.supabase.co")

    with pytest.raises(ValidationError):
        Settings(
            environment="test",
            supabase_url="https://project.supabase.co",
            supabase_publishable_key="sb_publishable_test",
            supabase_jwks_url="https://other.supabase.co/auth/v1/.well-known/jwks.json",
        )
