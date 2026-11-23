import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from ai_home.api import create_app
from ai_home.config import Settings
from ai_home.database import Base
from ai_home.models import TrainingContribution


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


def register_and_login(client: TestClient, email: str) -> str:
    password = "a-strong-password-123"
    registered = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password},
    )
    assert registered.status_code == 201
    login = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert login.status_code == 200
    return login.json()["access_token"]


def candidate_payload(**overrides) -> dict:
    payload = {
        "conversation_id": "d24b0b20-5ebc-48af-a3c8-000000000001",
        "provider": "openai",
        "source_endpoint": "https://api.openai.com/v1",
        "model": "gpt-test",
        "messages": [
            {"role": "user", "content": "Explain a public math concept."},
            {"role": "assistant", "content": "Here is a simple explanation."},
        ],
        "consent": True,
        "consent_version": "training_consent_v1",
        "consent_scope": "conversation_for_aihome_training_candidates",
        "provider_terms_allow_reuse": True,
        "content_reviewed_for_sensitive_data": True,
    }
    return {**payload, **overrides}


def test_training_contribution_requires_authentication_and_separate_attestations(
    client: TestClient,
):
    unauthenticated = client.post(
        "/api/v1/training/conversations",
        json=candidate_payload(),
    )
    token = register_and_login(client, "alice@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    no_consent = client.post(
        "/api/v1/training/conversations",
        headers=headers,
        json=candidate_payload(consent=False),
    )
    terms_denied = client.post(
        "/api/v1/training/conversations",
        headers=headers,
        json=candidate_payload(provider_terms_allow_reuse=False),
    )
    not_reviewed = client.post(
        "/api/v1/training/conversations",
        headers=headers,
        json=candidate_payload(content_reviewed_for_sensitive_data=False),
    )

    assert unauthenticated.status_code == 401
    assert no_consent.status_code == 422
    assert terms_denied.status_code == 422
    assert not_reviewed.status_code == 422


@pytest.mark.parametrize(
    ("provider", "endpoint"),
    [
        ("openai", "https://attacker.example/v1"),
        ("openai-compatible", "http://provider.example/v1"),
        ("openai-compatible", "https://user:secret@provider.example/v1"),
        ("openai-compatible", "https://provider.example/v1/sk-" + "x" * 24),
    ],
)
def test_training_provenance_rejects_spoofed_or_unsafe_endpoints(
    client: TestClient,
    provider: str,
    endpoint: str,
):
    token = register_and_login(client, "alice@example.com")
    response = client.post(
        "/api/v1/training/conversations",
        headers={"Authorization": f"Bearer {token}"},
        json=candidate_payload(provider=provider, source_endpoint=endpoint),
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "secret",
    [
        "Bearer " + "x" * 24,
        "sk-" + "x" * 24,
        "AIza" + "x" * 35,
        "-----BEGIN RSA PRIVATE KEY-----",
    ],
)
def test_training_candidate_rejects_common_secrets_without_persisting_them(
    client: TestClient,
    secret: str,
):
    token = register_and_login(client, "alice@example.com")
    response = client.post(
        "/api/v1/training/conversations",
        headers={"Authorization": f"Bearer {token}"},
        json=candidate_payload(
            messages=[
                {"role": "user", "content": f"Here is a credential: {secret}"},
                {"role": "assistant", "content": "Acknowledged."},
            ]
        ),
    )

    assert response.status_code == 422
    assert secret not in response.text
    with client.app.state.session_factory() as session:
        assert session.scalar(select(TrainingContribution)) is None


def test_contribution_records_provenance_and_can_be_withdrawn_before_freeze(
    client: TestClient,
):
    alice = register_and_login(client, "alice@example.com")
    bob = register_and_login(client, "bob@example.com")
    alice_headers = {"Authorization": f"Bearer {alice}"}
    bob_headers = {"Authorization": f"Bearer {bob}"}

    created = client.post(
        "/api/v1/training/conversations",
        headers=alice_headers,
        json=candidate_payload(),
    )
    contribution_id = created.json()["contribution_id"]

    assert created.status_code == 201
    assert created.headers["cache-control"] == "private, no-store"
    assert created.json()["provider"] == "openai"
    assert created.json()["source_endpoint"] == "https://api.openai.com/v1"
    assert created.json()["conversation_id"] == (
        "d24b0b20-5ebc-48af-a3c8-000000000001"
    )
    assert created.json()["model"] == "gpt-test"
    assert created.json()["consent_version"] == "training_consent_v1"
    assert created.json()["consent_scope"] == (
        "conversation_for_aihome_training_candidates"
    )
    assert created.json()["status"] == "pending_review"
    assert "messages" not in created.json()

    alice_list = client.get(
        "/api/v1/training/conversations",
        headers=alice_headers,
    )
    alice_data = client.get(
        f"/api/v1/training/conversations/{contribution_id}",
        headers=alice_headers,
    )
    bob_data = client.get(
        "/api/v1/training/conversations",
        headers=bob_headers,
    )
    bob_detail = client.get(
        f"/api/v1/training/conversations/{contribution_id}",
        headers=bob_headers,
    )
    cross_account_delete = client.delete(
        f"/api/v1/training/conversations/{contribution_id}",
        headers=bob_headers,
    )
    withdrawn = client.delete(
        f"/api/v1/training/conversations/{contribution_id}",
        headers=alice_headers,
    )
    after_withdrawal = client.get(
        "/api/v1/training/conversations",
        headers=alice_headers,
    )

    assert "messages" not in alice_list.json()[0]
    assert alice_data.json()["messages"] == candidate_payload()["messages"]
    assert bob_data.json() == []
    assert bob_detail.status_code == 404
    assert cross_account_delete.status_code == 404
    assert withdrawn.status_code == 204
    assert after_withdrawal.json() == []


def test_conversation_can_only_be_submitted_once_per_account(client: TestClient):
    token = register_and_login(client, "alice@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    first = client.post(
        "/api/v1/training/conversations",
        headers=headers,
        json=candidate_payload(),
    )
    duplicate = client.post(
        "/api/v1/training/conversations",
        headers=headers,
        json=candidate_payload(),
    )

    assert first.status_code == 201
    assert duplicate.status_code == 409


def test_training_contributions_can_be_paginated(client: TestClient):
    token = register_and_login(client, "alice@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    for suffix in ("0002", "0003"):
        response = client.post(
            "/api/v1/training/conversations",
            headers=headers,
            json=candidate_payload(
                conversation_id=f"d24b0b20-5ebc-48af-a3c8-00000000{suffix}"
            ),
        )
        assert response.status_code == 201

    first_page = client.get(
        "/api/v1/training/conversations?limit=1&offset=0",
        headers=headers,
    )
    second_page = client.get(
        "/api/v1/training/conversations?limit=1&offset=1",
        headers=headers,
    )

    assert len(first_page.json()) == len(second_page.json()) == 1
    assert (
        first_page.json()[0]["contribution_id"]
        != second_page.json()[0]["contribution_id"]
    )


def test_frozen_candidate_cannot_be_deleted_and_explains_irreversibility(
    client: TestClient,
):
    token = register_and_login(client, "alice@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    created = client.post(
        "/api/v1/training/conversations",
        headers=headers,
        json=candidate_payload(),
    )
    contribution_id = created.json()["contribution_id"]
    with client.app.state.session_factory() as session:
        contribution = session.get(TrainingContribution, contribution_id)
        assert contribution is not None
        contribution.status = "frozen"
        session.commit()

    response = client.delete(
        f"/api/v1/training/conversations/{contribution_id}",
        headers=headers,
    )

    assert response.status_code == 409
    assert "cannot be undone" in response.json()["detail"]
    assert client.get(
        "/api/v1/training/conversations",
        headers=headers,
    ).json()[0]["status"] == "frozen"


def test_account_deletion_also_removes_training_contributions(client: TestClient):
    token = register_and_login(client, "alice@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    submitted = client.post(
        "/api/v1/training/conversations",
        headers=headers,
        json=candidate_payload(),
    )
    assert submitted.status_code == 201

    deleted = client.request(
        "DELETE",
        "/api/v1/users/me",
        headers=headers,
        json={"password": "a-strong-password-123"},
    )

    assert deleted.status_code == 204
    with client.app.state.session_factory() as session:
        assert session.scalar(select(TrainingContribution)) is None
