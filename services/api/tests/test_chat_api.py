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
            chat_backend="openai-compatible",
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


def test_foundation_chat_uses_real_trained_checkpoint(client, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from ai_home.foundation.model import FoundationConfig
    from ai_home.foundation.training import TrainingConfig, train
    from ai_home.foundation.inference import complete

    torch.set_num_threads(1)
    records = [{"source": "authored API test", "url": "local:test", "license": "Apache-2.0",
                "date": "2026-11-26", "language": "es", "training_allowed": True,
                "privacy_reviewed": True, "visibility": "public",
                "text": (f"Documento {i}: entrenamos un modelo propio con datos revisados. " * 12)}
               for i in range(8)]
    path = tmp_path / "candidate.pt"
    result = train(records, path,
                   FoundationConfig(vocab_size=280, width=32, layers=2, heads=4,
                                    kv_heads=2, hidden=64, context=16),
                   TrainingConfig(steps=2, warmup=1), "cpu")
    assert result["embedding_delta_l2"] > 0
    settings = client.app.state.settings
    settings.chat_backend = "foundation"
    settings.foundation_checkpoint = path
    settings.foundation_device = "cpu"
    settings.foundation_max_new_tokens = 8

    def forbidden_external(*args, **kwargs):
        pytest.fail("Foundation must never fall back to external inference")

    monkeypatch.setattr("ai_home.api.chat_completion", forbidden_external)
    messages = [{"role": "user", "content": "Hola"}]
    assert client.post("/api/v1/chat/completions", json={"messages": messages}).status_code == 401
    token = create_session(client)
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/v1/chat/completions", headers=headers, json={"messages": messages})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json()["choices"][0]["message"]["content"] == complete(
        messages, path, device="cpu", max_new_tokens=8)
    # A subsequent conversation has its own prompt and cache.
    other = [{"role": "user", "content": "Otra pregunta"}]
    response = client.post("/api/v1/chat/completions", headers=headers, json={"messages": other})
    assert response.json()["choices"][0]["message"]["content"] == complete(
        other, path, device="cpu", max_new_tokens=8)
    from uuid import uuid4
    from sqlalchemy import select
    from ai_home.models import User
    from ai_home.credits import CreditLedger
    from ai_home.job_credits import user_account_id
    settings.foundation_inference_price = 7
    with client.app.state.session_factory() as session:
        owner = user_account_id(session.scalar(select(User.user_id)))
    ledger = CreditLedger(client.app.state.engine)
    ledger.create_account("test-chat-issuance", "test", allow_negative=True)
    ledger.create_account(owner, "user")
    ledger.post(idempotency_key="chat-funding", operation="test", postings={"test-chat-issuance": -20, owner: 20})
    assert client.get("/api/v1/credits/inference-price", headers=headers).json()["credits_per_completion"] == 7
    assert client.post("/api/v1/chat/completions", headers=headers, json={"messages": messages}).status_code == 402
    paid = {"messages": messages, "credit_limit": 7, "credit_request_id": str(uuid4())}
    response = client.post("/api/v1/chat/completions", headers=headers, json=paid)
    assert response.status_code == 200 and response.json()["charged_credits"] == 7
    assert response.json()["choices"][0]["message"]["content"] == complete(messages, path, device="cpu", max_new_tokens=8)
    assert client.post("/api/v1/chat/completions", headers=headers, json=paid).status_code == 409
    assert ledger.balance(owner) == 13
    assert ledger.balance("service:foundation-inference") == 7
    assert client.get("/api/v1/credits", headers=headers).json()["reserved"] == 0
    stream_duplicate = client.post("/api/v1/chat/stream", headers=headers, json=paid)
    assert '"code": "credit_request_conflict"' in stream_duplicate.text
    streamed = {**paid, "credit_request_id": str(uuid4())}
    stream_response = client.post("/api/v1/chat/stream", headers=headers, json=streamed)
    assert '"charged_credits": 7' in stream_response.text and '"type": "done"' in stream_response.text
    assert ledger.balance(owner) == 6
    assert ledger.balance("service:foundation-inference") == 14
    insufficient = client.post("/api/v1/chat/stream", headers=headers,
                               json={**paid, "credit_request_id": str(uuid4())})
    assert '"code": "insufficient_credits"' in insufficient.text
    ledger.post(idempotency_key="extra-chat-funding", operation="test", postings={"test-chat-issuance": -7, owner: 7})
    path.write_bytes(b"invalid checkpoint")
    failed = {**paid, "credit_request_id": str(uuid4())}
    assert client.post("/api/v1/chat/completions", headers=headers, json=failed).status_code == 503
    assert ledger.balance(owner) == 13
    assert client.get("/api/v1/credits", headers=headers).json()["reserved"] == 0
    assert client.post("/api/v1/chat/completions", headers=headers, json=failed).status_code == 409
    settings.foundation_inference_price = 0
    path.write_bytes(b"invalid checkpoint")
    response = client.post("/api/v1/chat/completions", headers=headers, json={"messages": messages})
    assert response.status_code == 503
    assert str(tmp_path) not in response.text
    path.unlink()
    assert client.post("/api/v1/chat/completions", headers=headers,
                       json={"messages": messages}).status_code == 503


def test_foundation_is_default_and_has_bounded_generation():
    assert Settings(_env_file=None).chat_backend == "foundation"
    for value in (0, 4097):
        with pytest.raises(ValueError):
            Settings(_env_file=None, foundation_max_new_tokens=value)

