import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from ai_home.api import create_app
from ai_home.config import Settings
from ai_home.database import Base
from ai_home.models import AuthSession, User


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    app = create_app(Settings(environment="test"), engine=engine)
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()


def register(client: TestClient, email: str = "alice@example.com") -> dict:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "a-strong-password-123"},
    )
    assert response.status_code == 201
    return response.json()


def login(client: TestClient, email: str = "alice@example.com") -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "a-strong-password-123"},
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    return response.json()["access_token"]


def test_registration_does_not_expose_password_or_hash(client: TestClient):
    user = register(client)
    assert user["email"] == "alice@example.com"
    assert "password" not in user
    assert "password_hash" not in user


def test_email_is_normalized_and_duplicate_registration_conflicts(client: TestClient):
    register(client, "Alice@Example.COM")
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "alice@example.com", "password": "a-strong-password-456"},
    )
    assert response.status_code == 409


def test_login_uses_same_error_for_unknown_user_and_wrong_password(client: TestClient):
    register(client)
    wrong_password = client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.com", "password": "wrong-password-123"},
    )
    unknown_user = client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "wrong-password-123"},
    )
    assert wrong_password.status_code == unknown_user.status_code == 401
    assert wrong_password.json() == unknown_user.json()


def test_private_profile_requires_session_and_returns_only_current_user(client: TestClient):
    register(client, "alice@example.com")
    bob = register(client, "bob@example.com")
    token = login(client, "alice@example.com")

    unauthenticated = client.get("/api/v1/users/me")
    alice = client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert unauthenticated.status_code == 401
    assert alice.status_code == 200
    assert alice.json()["email"] == "alice@example.com"
    assert alice.json()["user_id"] != bob["user_id"]
    assert "password_hash" not in alice.json()


def test_logout_revokes_bearer_token(client: TestClient):
    register(client)
    token = login(client)
    headers = {"Authorization": f"Bearer {token}"}

    response = client.post("/api/v1/auth/logout", headers=headers)
    after_logout = client.get("/api/v1/users/me", headers=headers)

    assert response.status_code == 204
    assert after_logout.status_code == 401


def test_password_and_bearer_token_are_only_stored_as_hashes(client: TestClient):
    register(client)
    token = login(client)
    session_factory = client.app.state.session_factory

    with session_factory() as session:
        user = session.scalar(select(User).where(User.email == "alice@example.com"))
        auth_session = session.scalar(select(AuthSession))

    assert user is not None
    assert user.password_hash.startswith("$argon2id$")
    assert "a-strong-password-123" not in user.password_hash
    assert auth_session is not None
    assert auth_session.token_hash != token


def test_account_export_excludes_credentials(client: TestClient):
    register(client)
    token = login(client)
    response = client.get(
        "/api/v1/users/me/export",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["email"] == "alice@example.com"
    assert "password_hash" not in response.json()


def test_account_deletion_requires_password_and_revokes_all_sessions(
    client: TestClient,
):
    register(client)
    first_token = login(client)
    second_token = login(client)
    headers = {"Authorization": f"Bearer {first_token}"}

    rejected = client.request(
        "DELETE",
        "/api/v1/users/me",
        headers=headers,
        json={"password": "a-different-password-999"},
    )
    deleted = client.request(
        "DELETE",
        "/api/v1/users/me",
        headers=headers,
        json={"password": "a-strong-password-123"},
    )
    first_session_after_delete = client.get("/api/v1/users/me", headers=headers)
    second_session_after_delete = client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {second_token}"},
    )

    assert rejected.status_code == 403
    assert deleted.status_code == 204
    assert first_session_after_delete.status_code == 401
    assert second_session_after_delete.status_code == 401


def test_database_readiness_endpoint_checks_database(client: TestClient):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": "ok"}


def test_registration_rejects_short_password(client: TestClient):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "alice@example.com", "password": "short"},
    )
    assert response.status_code == 422



def test_login_identity_is_stable_and_matches_authenticated_profile(client):
    user = register(client)
    for _ in range(2):
        response = client.post("/api/v1/auth/login", json={
            "email": "alice@example.com", "password": "a-strong-password-123",
        })
        assert response.status_code == 200
        assert response.json()["user_id"] == user["user_id"]
