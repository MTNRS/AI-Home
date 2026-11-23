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
    )
    assert settings.environment == "production"


def test_session_lifetime_has_a_safe_minimum():
    with pytest.raises(ValidationError):
        Settings(environment="test", session_lifetime_seconds=30)
