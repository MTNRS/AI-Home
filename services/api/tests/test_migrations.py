from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import URL


def test_initial_migration_creates_identity_and_credit_tables(tmp_path, monkeypatch):
    database_path = tmp_path / "migration-test.db"
    database_url = URL.create(
        "sqlite+pysqlite",
        database=str(database_path),
    )
    monkeypatch.setenv("AI_HOME_ENVIRONMENT", "test")
    monkeypatch.setenv("AI_HOME_DATABASE_URL", str(database_url))

    config = Config("alembic.ini")
    command.upgrade(config, "head")
    command.check(config)

    engine = create_engine(database_url)
    try:
        table_names = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert {
        "users",
        "auth_sessions",
        "credit_accounts",
        "credit_transactions",
        "credit_postings",
        "compute_nodes",
        "compute_jobs",
        "alembic_version",
    } <= table_names
    command.check(config)

    command.downgrade(config, "base")
    engine = create_engine(database_url)
    try:
        table_names = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert not {
        "users",
        "auth_sessions",
        "credit_accounts",
        "credit_transactions",
        "credit_postings",
        "compute_nodes",
        "compute_jobs",
    } & table_names
