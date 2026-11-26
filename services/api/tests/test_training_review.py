import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from ai_home.api import create_app
from ai_home.config import Settings
from ai_home.database import Base
from ai_home.models import TrainingContribution, TrainingDataset

REVIEW_TOKEN = "test-training-review-token-0123456789"


def test_frozen_contributions_reach_real_sft_without_public_release(client, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from ai_home.foundation.contributions import load_reviewed_export
    from ai_home.foundation.data import canonical_hash, prepare_records
    from ai_home.foundation.model import FoundationConfig
    from ai_home.foundation.training import TrainingConfig, train
    from ai_home.foundation.checkpoint import load_checkpoint

    torch.set_num_threads(1)
    token = register_and_login(client)
    ids = [create_candidate(client, token, str(i)) for i in range(1, 5)]
    for contribution_id in ids:
        approve_candidate(client, contribution_id)
    headers = {"Authorization": f"Bearer {REVIEW_TOKEN}"}
    frozen = client.post("/api/v1/admin/training/datasets", headers=headers,
                         json={"dataset_version": "sft-test-v1", "contribution_ids": ids})
    assert frozen.status_code == 201
    route = "/api/v1/admin/training/datasets/" + frozen.json()["dataset_id"]
    assert client.get(route, headers={"Authorization": f"Bearer {token}"}).status_code == 401
    exported = client.get(route, headers=headers).json()
    path = tmp_path / "reviewed.json"
    path.write_text(json.dumps(exported), encoding="utf-8")
    records = load_reviewed_export(path)
    assert all(r["visibility"] == "reviewed-training-only" for r in records)
    with pytest.raises(ValueError, match="public"):
        prepare_records(records)
    output = tmp_path / "sft.pt"
    settings = TrainingConfig(steps=2, warmup=1, objective="sft")
    result = train(None, output, FoundationConfig(vocab_size=280, width=32, layers=2,
                   heads=4, kv_heads=2, hidden=64, context=16), settings, "cpu", reviewed_export=path)
    assert result["embedding_delta_l2"] > 0
    assert result["validation"]["tokens"] > 0
    assert load_checkpoint(output)[2]["data_visibility"] == "reviewed-training-only"
    assert result["status"] == "candidate"
    from dataclasses import asdict
    from ai_home.foundation.training_job import run_training
    model_path, training_path = tmp_path / "model.json", tmp_path / "training.json"
    model_path.write_text(json.dumps(result["config"]))
    training_path.write_text(json.dumps(asdict(settings)))
    job = run_training(path, model_path, training_path, tmp_path / "private-job", reviewed_export=True)
    assert load_checkpoint(job["checkpoint"])[2]["data_visibility"] == "reviewed-training-only"
    child = run_training(path, None, training_path, tmp_path / "private-child", reviewed_export=True,
                         initial_checkpoint=job["checkpoint"])
    assert child["embedding_delta_l2"] > 0
    assert load_checkpoint(child["checkpoint"])[2]["data_visibility"] == "reviewed-training-only"
    manifest = json.loads((tmp_path / "private-child" / "configuration.json").read_text())
    assert manifest["source_type"] == "reviewed-training-only"
    training_path.write_text(json.dumps(asdict(TrainingConfig(steps=2, warmup=1))))
    with pytest.raises(ValueError, match="sft"):
        run_training(path, model_path, training_path, tmp_path / "bad-objective", reviewed_export=True)
    assert not (tmp_path / "bad-objective").exists()
    # Replace the source with another internally valid export between both reads.
    # A manifest-valid replacement must not train under the first dataset hash.
    import copy
    from ai_home.foundation import training_job
    replacement = copy.deepcopy(exported)
    replacement["records"][0]["messages"][1]["content"] = "A different reviewed answer"
    replacement["manifest_sha256"] = canonical_hash({"dataset_version": replacement["dataset_version"],
                                                     "records": replacement["records"]})
    original_train = training_job.train
    def replace_before_training(*args, **kwargs):
        path.write_text(json.dumps(replacement), encoding="utf-8")
        return original_train(*args, **kwargs)
    training_path.write_text(json.dumps(asdict(settings)), encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(training_job, "train", replace_before_training)
        with pytest.raises(ValueError, match="changed after job configuration"):
            run_training(path, model_path, training_path, tmp_path / "changed-export", reviewed_export=True)
    assert not (tmp_path / "changed-export" / "model.pt").exists()
    assert not (tmp_path / "changed-export" / "result.json").exists()
    path.write_text(json.dumps(exported), encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds input budget"):
        load_reviewed_export(path, max_bytes=32)
    with pytest.raises(ValueError, match="Invalid reviewed export read budget"):
        load_reviewed_export(path, max_bytes=True)
    exported["records"][0]["messages"][1]["content"] = "tampered"
    path.write_text(json.dumps(exported), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        load_reviewed_export(path)
    exported["records"][0]["review_checklist"] = {}
    exported["manifest_sha256"] = canonical_hash({"dataset_version": exported["dataset_version"],
                                                 "records": exported["records"]})
    path.write_text(json.dumps(exported), encoding="utf-8")
    with pytest.raises(ValueError, match="review"):
        load_reviewed_export(path)


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
            training_review_token=REVIEW_TOKEN,
            training_reviewer_id="test-reviewer",
        ),
        engine=engine,
    )
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()


def register_and_login(client: TestClient) -> str:
    password = "a-strong-password-123"
    registered = client.post(
        "/api/v1/auth/register",
        json={"email": "alice@example.com", "password": password},
    )
    assert registered.status_code == 201
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.com", "password": password},
    )
    assert login.status_code == 200
    return login.json()["access_token"]


def create_candidate(client: TestClient, token: str, suffix: str) -> str:
    # Fixed contribution identities make hash-based family partitions reproducible.
    from unittest.mock import patch
    from uuid import UUID
    with patch("ai_home.api.uuid.uuid4", return_value=UUID(int=int(suffix))):
        response = client.post(
            "/api/v1/training/conversations",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "conversation_id": f"d24b0b20-5ebc-48af-a3c8-{suffix:0>12}",
                "provider": "openai",
                "source_endpoint": "https://api.openai.com/v1",
                "model": "gpt-test",
                "messages": [
                    {"role": "user", "content": f"Question {suffix}"},
                    {"role": "assistant", "content": f"Answer {suffix}"},
                ],
                "consent": True,
                "consent_version": "training_consent_v1",
                "consent_scope": "conversation_for_aihome_training_candidates",
                "provider_terms_allow_reuse": True,
                "content_reviewed_for_sensitive_data": True,
            },
        )
    assert response.status_code == 201
    return response.json()["contribution_id"]


def approve_candidate(client: TestClient, contribution_id: str) -> None:
    response = client.post(
        f"/api/v1/admin/training/conversations/{contribution_id}/review",
        headers={"Authorization": f"Bearer {REVIEW_TOKEN}"},
        json={
            "approve": True,
            "provider_terms_verified": True,
            "sensitive_data_reviewed": True,
            "reuse_rights_verified": True,
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "eligible"


def test_review_api_is_disabled_without_configuration_and_rejects_bad_tokens(
    client: TestClient,
):
    unconfigured = create_app(
        Settings(environment="test"),
        engine=client.app.state.engine,
    )
    with TestClient(unconfigured) as second_client:
        disabled = second_client.get("/api/v1/admin/training/conversations")
    invalid = client.get(
        "/api/v1/admin/training/conversations",
        headers={"Authorization": "Bearer " + "x" * 36},
    )

    assert disabled.status_code == 503
    assert invalid.status_code == 401
    assert REVIEW_TOKEN not in invalid.text


def test_review_requires_checklist_and_rejection_scrubs_conversation(
    client: TestClient,
):
    user_token = register_and_login(client)
    contribution_id = create_candidate(client, user_token, "1")
    headers = {"Authorization": f"Bearer {REVIEW_TOKEN}"}

    pending = client.get(
        "/api/v1/admin/training/conversations",
        headers=headers,
    )
    incomplete_approval = client.post(
        f"/api/v1/admin/training/conversations/{contribution_id}/review",
        headers=headers,
        json={"approve": True},
    )
    rejected = client.post(
        f"/api/v1/admin/training/conversations/{contribution_id}/review",
        headers=headers,
        json={"approve": False},
    )
    user_detail = client.get(
        f"/api/v1/training/conversations/{contribution_id}",
        headers={"Authorization": f"Bearer {user_token}"},
    )

    assert pending.status_code == 200
    assert pending.json()[0]["messages"][0]["content"] == "Question 1"
    assert incomplete_approval.status_code == 422
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    assert rejected.json()["reviewer_id"] == "test-reviewer"
    assert rejected.json()["review_checklist"] == {
        "provider_terms_verified": False,
        "sensitive_data_reviewed": False,
        "reuse_rights_verified": False,
    }
    assert user_detail.status_code == 200
    assert user_detail.json()["messages"] == []


def test_only_human_reviewed_candidates_can_be_frozen_into_versioned_dataset(
    client: TestClient,
):
    user_token = register_and_login(client)
    first_id = create_candidate(client, user_token, "1")
    second_id = create_candidate(client, user_token, "2")
    reviewer_headers = {"Authorization": f"Bearer {REVIEW_TOKEN}"}

    pending_freeze = client.post(
        "/api/v1/admin/training/datasets",
        headers=reviewer_headers,
        json={
            "dataset_version": "candidate-v1",
            "contribution_ids": [first_id],
        },
    )
    approve_candidate(client, first_id)
    approve_candidate(client, second_id)
    frozen = client.post(
        "/api/v1/admin/training/datasets",
        headers=reviewer_headers,
        json={
            "dataset_version": "candidate-v1",
            "contribution_ids": [first_id, second_id],
        },
    )

    assert pending_freeze.status_code == 409
    assert frozen.status_code == 201
    assert frozen.json()["dataset_version"] == "candidate-v1"
    assert frozen.json()["item_count"] == 2
    assert len(frozen.json()["manifest_sha256"]) == 64

    dataset = client.get(
        f"/api/v1/admin/training/datasets/{frozen.json()['dataset_id']}",
        headers=reviewer_headers,
    )
    expected_hash = hashlib.sha256(
        json.dumps(
            {
                "dataset_version": "candidate-v1",
                "records": dataset.json()["records"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    withdrawn = client.delete(
        f"/api/v1/training/conversations/{first_id}",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    duplicate_version = client.post(
        "/api/v1/admin/training/datasets",
        headers=reviewer_headers,
        json={
            "dataset_version": "candidate-v1",
            "contribution_ids": [first_id],
        },
    )

    assert dataset.status_code == 200
    assert dataset.json()["records"][0]["review_checklist"] == {
        "provider_terms_verified": True,
        "sensitive_data_reviewed": True,
        "reuse_rights_verified": True,
    }
    assert dataset.json()["manifest_sha256"] == expected_hash
    assert withdrawn.status_code == 409
    assert duplicate_version.status_code == 409
    with client.app.state.session_factory() as session:
        stored_dataset = session.scalar(select(TrainingDataset))
        frozen_contribution = session.get(TrainingContribution, first_id)
    assert stored_dataset is not None
    assert frozen_contribution is not None
    assert frozen_contribution.status == "frozen"


def test_dataset_freeze_validates_version_and_contribution_ids(client: TestClient):
    token = register_and_login(client)
    pending_id = create_candidate(client, token, "1")
    headers = {"Authorization": f"Bearer {REVIEW_TOKEN}"}

    invalid_version = client.post(
        "/api/v1/admin/training/datasets",
        headers=headers,
        json={"dataset_version": "../escape", "contribution_ids": [pending_id]},
    )
    missing_record = client.post(
        "/api/v1/admin/training/datasets",
        headers=headers,
        json={
            "dataset_version": "valid-v1",
            "contribution_ids": ["4d5871c5-4766-4813-b8b5-000000000001"],
        },
    )

    assert invalid_version.status_code == 422
    assert missing_record.status_code == 404


def test_training_reviewer_token_requires_minimum_length():
    with pytest.raises(ValueError, match="32 ASCII characters"):
        Settings(environment="test", training_review_token="short")


def test_training_reviewer_token_must_be_ascii():
    with pytest.raises(ValueError, match="32 ASCII characters"):
        Settings(environment="test", training_review_token="Ã©" * 32)


def test_account_deletion_does_not_rewrite_frozen_dataset_snapshot(
    client: TestClient,
):
    user_token = register_and_login(client)
    contribution_id = create_candidate(client, user_token, "1")
    approve_candidate(client, contribution_id)
    reviewer_headers = {"Authorization": f"Bearer {REVIEW_TOKEN}"}
    frozen = client.post(
        "/api/v1/admin/training/datasets",
        headers=reviewer_headers,
        json={
            "dataset_version": "frozen-v1",
            "contribution_ids": [contribution_id],
        },
    )
    assert frozen.status_code == 201

    deleted = client.request(
        "DELETE",
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {user_token}"},
        json={"password": "a-strong-password-123"},
    )
    dataset = client.get(
        f"/api/v1/admin/training/datasets/{frozen.json()['dataset_id']}",
        headers=reviewer_headers,
    )

    assert deleted.status_code == 204
    assert dataset.status_code == 200
    assert dataset.json()["records"][0]["messages"][0]["content"] == "Question 1"
    with client.app.state.session_factory() as session:
        assert session.get(TrainingContribution, contribution_id) is None

