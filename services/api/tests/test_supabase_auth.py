import json
import time
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
import httpx
import jwt
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ai_home.api import create_app
from ai_home.auth import create_supabase_user
from ai_home.config import Settings
from ai_home.database import Base
from ai_home.models import AuthSession, User
from ai_home.supabase_auth import InvalidSupabaseToken, SupabaseAuth


PROJECT_URL = "https://project.supabase.co"
PUBLISHABLE_KEY = "sb_publishable_test"
SECRET_KEY = "sb_secret_test"
USER_ID = "b26dcb5c-b4c7-4e29-a661-e264a82238e1"
USER_EMAIL = "alice@example.com"


def _signing_material():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(
        private_key.public_key(),
        as_dict=True,
    )
    public_jwk.update({"kid": "test-key", "alg": "RS256", "use": "sig"})
    access_token = jwt.encode(
        {
            "iss": f"{PROJECT_URL}/auth/v1",
            "aud": "authenticated",
            "sub": USER_ID,
            "role": "authenticated",
            "email": USER_EMAIL,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-key"},
    )
    return public_jwk, access_token


def _settings() -> Settings:
    return Settings(
        environment="test",
        supabase_url=PROJECT_URL,
        supabase_publishable_key=PUBLISHABLE_KEY,
        supabase_secret_key=SECRET_KEY,
    )


def _api_client(handler):
    transport = httpx.MockTransport(handler)
    auth_client = SupabaseAuth(
        project_url=PROJECT_URL,
        publishable_key=PUBLISHABLE_KEY,
        secret_key=SECRET_KEY,
        jwks_url=f"{PROJECT_URL}/auth/v1/.well-known/jwks.json",
        transport=transport,
    )
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    app = create_app(_settings(), engine=engine, supabase_auth=auth_client)
    return TestClient(app), engine


def test_supabase_auth_registration_login_token_verification_and_logout():
    signing_key, access_token = _signing_material()
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/.well-known/jwks.json"):
            return httpx.Response(200, json={"keys": [signing_key]})
        if request.url.path.endswith("/signup"):
            body = json.loads(request.read())
            assert body == {
                "email": USER_EMAIL,
                "password": "a-strong-password-123",
            }
            assert request.headers["apikey"] == PUBLISHABLE_KEY
            return httpx.Response(
                200,
                json={"id": USER_ID, "email": USER_EMAIL},
            )
        if request.url.path.endswith("/token"):
            body = json.loads(request.read())
            assert body == {
                "email": USER_EMAIL,
                "password": "a-strong-password-123",
            }
            return httpx.Response(
                200,
                json={
                    "access_token": access_token,
                    "expires_in": 3600,
                    "user": {"id": USER_ID, "email": USER_EMAIL},
                },
            )
        if request.url.path.endswith("/logout"):
            assert request.headers["Authorization"] == f"Bearer {access_token}"
            return httpx.Response(204)
        raise AssertionError(f"Unexpected Supabase request: {request.url}")

    client, engine = _api_client(handle)
    try:
        registered = client.post(
            "/api/v1/auth/register",
            json={"email": USER_EMAIL, "password": "a-strong-password-123"},
        )
        login = client.post(
            "/api/v1/auth/login",
            json={"email": USER_EMAIL, "password": "a-strong-password-123"},
        )
        headers = {"Authorization": f"Bearer {access_token}"}
        profile = client.get("/api/v1/users/me", headers=headers)
        logged_out = client.post("/api/v1/auth/logout", headers=headers)
        rejected_after_logout = client.get("/api/v1/users/me", headers=headers)

        assert registered.status_code == 201
        assert registered.json()["user_id"] == USER_ID
        assert "password_hash" not in registered.json()
        assert login.status_code == 200
        assert login.json()["access_token"] == access_token
        assert profile.status_code == 200
        assert profile.json()["email"] == USER_EMAIL
        assert logged_out.status_code == 204
        assert rejected_after_logout.status_code == 401

        with client.app.state.session_factory() as session:
            user = session.get(User, USER_ID)
            auth_session = session.scalar(select(AuthSession))
        assert user is not None and user.password_hash is None
        assert auth_session is not None
        assert auth_session.token_hash != access_token
        assert len(calls) == 4
    finally:
        client.close()
        engine.dispose()


def test_supabase_account_deletion_uses_server_secret_and_removes_profile():
    signing_key, access_token = _signing_material()

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/.well-known/jwks.json"):
            return httpx.Response(200, json={"keys": [signing_key]})
        if request.url.path.endswith("/signup"):
            return httpx.Response(
                200,
                json={"id": USER_ID, "email": USER_EMAIL},
            )
        if request.url.path.endswith("/token"):
            assert json.loads(request.read())["password"] == (
                "a-strong-password-123"
            )
            return httpx.Response(
                200,
                json={
                    "access_token": access_token,
                    "expires_at": int(time.time()) + 3600,
                    "user": {"id": USER_ID, "email": USER_EMAIL},
                },
            )
        if request.method == "DELETE":
            assert request.url.path.endswith(f"/admin/users/{USER_ID}")
            assert request.headers["apikey"] == SECRET_KEY
            assert request.headers["Authorization"] == f"Bearer {SECRET_KEY}"
            return httpx.Response(200, json={"id": USER_ID})
        raise AssertionError(f"Unexpected Supabase request: {request.url}")

    client, engine = _api_client(handle)
    try:
        client.post(
            "/api/v1/auth/register",
            json={"email": USER_EMAIL, "password": "a-strong-password-123"},
        )
        client.post(
            "/api/v1/auth/login",
            json={"email": USER_EMAIL, "password": "a-strong-password-123"},
        )
        deleted = client.request(
            "DELETE",
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"password": "a-strong-password-123"},
        )

        assert deleted.status_code == 204
        assert client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {access_token}"},
        ).status_code == 401
        with client.app.state.session_factory() as session:
            assert session.get(User, USER_ID) is None
    finally:
        client.close()
        engine.dispose()


def test_supabase_token_rejects_invalid_issuer():
    signing_key, _ = _signing_material()
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    wrong_issuer_token = jwt.encode(
        {
            "iss": "https://other-project.supabase.co/auth/v1",
            "aud": "authenticated",
            "sub": USER_ID,
            "role": "authenticated",
            "email": USER_EMAIL,
            "exp": int(time.time()) + 3600,
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-key"},
    )

    auth_client = SupabaseAuth(
        project_url=PROJECT_URL,
        publishable_key=PUBLISHABLE_KEY,
        jwks_url=f"{PROJECT_URL}/auth/v1/.well-known/jwks.json",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"keys": [signing_key]})
        ),
    )

    try:
        auth_client.verify_access_token(wrong_issuer_token)
    except InvalidSupabaseToken:
        pass
    else:
        raise AssertionError("Tokens from another Supabase project must be rejected.")


def test_supabase_login_explains_when_email_confirmation_is_required():
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/token")
        return httpx.Response(
            400,
            json={
                "code": "email_not_confirmed",
                "message": "Email not confirmed",
            },
        )

    client, engine = _api_client(handle)
    try:
        response = client.post(
            "/api/v1/auth/login",
            json={"email": USER_EMAIL, "password": "a-strong-password-123"},
        )

        assert response.status_code == 403
        assert response.json()["detail"] == (
            "Confirma tu correo electrónico antes de iniciar sesión."
        )
    finally:
        client.close()
        engine.dispose()


def test_supabase_user_projection_does_not_store_a_password():
    engine = create_engine("sqlite+pysqlite://")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session, session.begin():
            user = create_supabase_user(
                session,
                user_id=USER_ID,
                email=USER_EMAIL,
            )
            assert user.user_id == USER_ID
            assert user.password_hash is None
    finally:
        engine.dispose()

