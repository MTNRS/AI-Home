import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from ai_home.api import create_app
from ai_home.config import Settings
from ai_home.database import Base


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    app = create_app(
        Settings(
            environment="test",
            chat_provider_url="http://localhost:11434/v1",
            chat_provider_model="test-model",
            chat_provider_api_key="provider-secret",
        ),
        engine=engine,
    )
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()


def create_session(client: TestClient) -> str:
    email = "alice@example.com"
    password = "a-strong-password-123"
    registered = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password},
    )
    login = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert registered.status_code == 201
    assert login.status_code == 200
    return login.json()["access_token"]


def test_chat_settings_require_model_and_reject_provider_credentials_in_url():
    with pytest.raises(ValueError):
        Settings(
            environment="test",
            chat_provider_url="http://localhost:11434/v1",
        )
    with pytest.raises(ValueError):
        Settings(
            environment="test",
            chat_provider_url="https://user:secret@provider.example/v1",
            chat_provider_model="test-model",
        )


def test_chat_endpoint_requires_account_and_uses_server_configured_provider(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    observed = {}

    def fake_completion(endpoint_url, model, messages, *, api_key, timeout_seconds):
        observed.update(
            endpoint_url=endpoint_url,
            model=model,
            messages=messages,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        return "Respuesta de prueba."

    monkeypatch.setattr("ai_home.api.chat_completion", fake_completion)
    payload = {"messages": [{"role": "user", "content": "Hola"}]}
    unauthenticated = client.post("/api/v1/chat/completions", json=payload)

    token = create_session(client)
    response = client.post(
        "/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )

    assert unauthenticated.status_code == 401
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Respuesta de prueba.",
    }
    assert observed == {
        "endpoint_url": "http://localhost:11434/v1",
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hola"}],
        "api_key": "provider-secret",
        "timeout_seconds": 180,
    }
    assert "provider-secret" not in response.text


def test_chat_endpoint_is_unavailable_without_provider_configuration(
    client: TestClient,
):
    client.app.state.settings.chat_provider_url = None
    token = create_session(client)

    response = client.post(
        "/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={"messages": [{"role": "user", "content": "Hola"}]},
    )

    assert response.status_code == 503


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "No hay mensaje de usuario."}],
        [{"role": "user", "content": "x" * 8193}],
        [{"role": "user", "content": "x" * 8000}] * 5,
        [{"role": "user", "content": "ok", "unexpected": True}],
    ],
)
def test_chat_endpoint_rejects_invalid_or_unbounded_messages(
    client: TestClient,
    messages,
):
    token = create_session(client)

    response = client.post(
        "/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={"messages": messages},
    )

    assert response.status_code == 422


def test_chat_endpoint_converts_provider_failure_to_gateway_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    def failed_completion(*_args, **_kwargs):
        raise httpx.ConnectError("provider unavailable")

    monkeypatch.setattr("ai_home.api.chat_completion", failed_completion)
    token = create_session(client)

    response = client.post(
        "/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={"messages": [{"role": "user", "content": "Hola"}]},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Configured chat provider failed to complete the request."
    )

