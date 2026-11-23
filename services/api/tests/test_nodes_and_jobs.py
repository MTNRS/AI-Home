import sqlite3

import pytest
import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

import ai_home.jobs as job_service
from ai_home.api import create_app
from ai_home.config import Settings
from ai_home.database import Base
from ai_home.jobs import JOB_LEASE_SECONDS
from ai_home.models import ComputeJob, ComputeNode, User
from ai_home.node_worker import (
    NodeAgent,
    _parse_nvidia_smi,
    execute_job,
    register_node_with_server,
)
from ai_home.schemas import HardwareReport


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


def register(client: TestClient, email: str) -> dict:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "a-strong-password-123"},
    )
    assert response.status_code == 201
    assert response.headers["cache-control"] == "private, no-store"
    return response.json()


def login(client: TestClient, email: str) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "a-strong-password-123"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def user_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def hardware_report() -> dict:
    return {
        "platform": "test",
        "cpu_model": "Test CPU",
        "logical_cores": 4,
        "ram_total_bytes": 8 * 1024**3,
        "gpus": [
            {
                "device_index": 0,
                "name": "Test GPU",
                "memory_total_bytes": 4 * 1024**3,
            },
            {
                "device_index": 1,
                "name": "Second Test GPU",
                "memory_total_bytes": 2 * 1024**3,
            },
        ],
    }


def create_node(client: TestClient, token: str, name: str = "node-one") -> dict:
    response = client.post(
        "/api/v1/nodes",
        headers=user_headers(token),
        json={"name": name, "hardware": hardware_report()},
    )
    assert response.status_code == 201
    assert response.headers["cache-control"] == "private, no-store"
    return response.json()


def create_job(
    client: TestClient,
    token: str,
    *,
    values: list[int] | None = None,
    key: str = "job-1",
    consent: bool = True,
):
    return client.post(
        "/api/v1/jobs",
        headers=user_headers(token),
        json={
            "values": values if values is not None else [7, -2, 15],
            "idempotency_key": key,
            "allow_public_node_processing": consent,
        },
    )


def test_node_registration_returns_one_time_secret_and_separate_devices(
    client: TestClient,
):
    register(client, "alice@example.com")
    alice_token = login(client, "alice@example.com")
    created = create_node(client, alice_token)
    stored_secret = None

    with client.app.state.session_factory() as session:
        stored = session.scalar(
            select(ComputeNode).where(ComputeNode.node_id == created["node_id"])
        )
        stored_secret = stored.credential_hash if stored else None

    assert created["node_token"]
    assert len(created["supported_job_types"]) == 1
    assert created["supported_job_types"] == ["math.sum.v1"]
    assert stored_secret != created["node_token"]

    listed = client.get("/api/v1/nodes", headers=user_headers(alice_token))
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert len(listed.json()[0]["hardware"]["gpus"]) == 2
    assert "credential_hash" not in listed.text
    assert listed.headers["cache-control"] == "private, no-store"


def test_jobs_require_explicit_consent_to_share_with_nodes(client: TestClient):
    register(client, "alice@example.com")
    alice_token = login(client, "alice@example.com")

    denied = create_job(client, alice_token, consent=False)
    accepted = create_job(client, alice_token)

    assert denied.status_code == 422
    assert accepted.status_code == 201
    assert accepted.json()["status"] == "queued"
    with client.app.state.session_factory() as session:
        stored_job = session.get(ComputeJob, accepted.json()["job_id"])
    assert stored_job is not None
    assert stored_job.public_node_consent_at_epoch > 0
    assert stored_job.public_node_consent_version == "volunteer-compute-v1"


def test_job_idempotency_returns_existing_and_rejects_changed_payload(
    client: TestClient,
):
    register(client, "alice@example.com")
    token = login(client, "alice@example.com")

    first = create_job(client, token, key="same-key", values=[3, 5])
    repeated = create_job(client, token, key="same-key", values=[3, 5])
    conflicting = create_job(client, token, key="same-key", values=[3, 6])

    assert repeated.status_code == 201
    assert repeated.json()["job_id"] == first.json()["job_id"]
    assert conflicting.status_code == 409


def test_concurrent_idempotency_insert_recovers_existing_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    register(client, "alice@example.com")
    token = login(client, "alice@example.com")
    first = create_job(client, token, key="race-key", values=[3, 5])
    unique_violation = IntegrityError(
        "insert",
        {},
        sqlite3.IntegrityError(
            "UNIQUE constraint failed: compute_jobs.owner_user_id, "
            "compute_jobs.idempotency_key"
        ),
    )

    def raise_unique_violation(*args, **kwargs):
        raise unique_violation

    monkeypatch.setattr(
        "ai_home.api.create_math_sum_job",
        raise_unique_violation,
    )

    retry = create_job(client, token, key="race-key", values=[3, 5])
    conflict = create_job(client, token, key="race-key", values=[3, 6])

    assert retry.status_code == 201
    assert retry.json()["job_id"] == first.json()["job_id"]
    assert conflict.status_code == 409


def test_node_executes_allowlisted_job_and_server_validates_result(
    client: TestClient,
):
    register(client, "alice@example.com")
    alice_token = login(client, "alice@example.com")
    node = create_node(client, alice_token)
    job = create_job(client, alice_token, values=[100, -25, 9]).json()
    node_headers = user_headers(node["node_token"])

    heartbeat = client.post("/api/v1/node/heartbeat", headers=node_headers)
    claimed = client.post("/api/v1/node/jobs/claim", headers=node_headers)

    assert heartbeat.status_code == 200
    assert claimed.status_code == 200
    claim = claimed.json()
    assert claim["job_id"] == job["job_id"]
    assert claim["lease_expires_epoch"] > 0
    result = execute_job(claim["job_type"], claim["payload"])
    submitted = client.post(
        f"/api/v1/node/jobs/{claim['job_id']}/result",
        headers=node_headers,
        json={"lease_token": claim["lease_token"], "result": result},
    )

    assert submitted.status_code == 200
    assert submitted.json()["status"] == "completed"
    assert submitted.json()["result"] == {"sum": 84}


def test_server_rejects_incorrect_node_result_without_completing_job(
    client: TestClient,
):
    register(client, "alice@example.com")
    token = login(client, "alice@example.com")
    node = create_node(client, token)
    create_job(client, token, values=[2, 3])
    headers = user_headers(node["node_token"])
    claim = client.post("/api/v1/node/jobs/claim", headers=headers).json()

    invalid = client.post(
        f"/api/v1/node/jobs/{claim['job_id']}/result",
        headers=headers,
        json={"lease_token": claim["lease_token"], "result": {"sum": 999}},
    )
    owner_jobs = client.get("/api/v1/jobs", headers=user_headers(token)).json()

    assert invalid.status_code == 422
    assert owner_jobs[0]["status"] == "leased"


def test_jobs_and_nodes_are_isolated_between_users(client: TestClient):
    register(client, "alice@example.com")
    register(client, "bob@example.com")
    alice_token = login(client, "alice@example.com")
    bob_token = login(client, "bob@example.com")
    alice_node = create_node(client, alice_token, "alice-node")
    create_node(client, bob_token, "bob-node")
    alice_job = create_job(client, alice_token, values=[11]).json()
    bob_job = create_job(client, bob_token, values=[22]).json()

    alice_nodes = client.get("/api/v1/nodes", headers=user_headers(alice_token))
    bob_jobs = client.get("/api/v1/jobs", headers=user_headers(bob_token))
    bob_claim = client.post(
        "/api/v1/node/jobs/claim",
        headers=user_headers(alice_node["node_token"]),
    )

    assert len(alice_nodes.json()) == 1
    assert [job["job_id"] for job in bob_jobs.json()] == [bob_job["job_id"]]
    assert bob_claim.status_code == 200
    assert bob_claim.json()["job_id"] in {alice_job["job_id"], bob_job["job_id"]}


def test_revoking_node_requeues_leased_job_and_invalidates_credential(
    client: TestClient,
):
    register(client, "alice@example.com")
    token = login(client, "alice@example.com")
    node = create_node(client, token)
    create_job(client, token)
    node_headers = user_headers(node["node_token"])
    claimed = client.post("/api/v1/node/jobs/claim", headers=node_headers)
    assert claimed.status_code == 200

    revoked = client.delete(
        f"/api/v1/nodes/{node['node_id']}",
        headers=user_headers(token),
    )
    after_revoke = client.post("/api/v1/node/heartbeat", headers=node_headers)
    job = client.get("/api/v1/jobs", headers=user_headers(token)).json()[0]

    assert revoked.status_code == 204
    assert after_revoke.status_code == 401
    assert job["status"] == "queued"


def test_expired_leases_are_reassigned_and_exhausted_jobs_fail(
    client: TestClient,
    monkeypatch,
):
    register(client, "alice@example.com")
    token = login(client, "alice@example.com")
    node = create_node(client, token)
    create_job(client, token)
    node_headers = user_headers(node["node_token"])
    now = int(job_service.time.time())
    current_time = [now]
    monkeypatch.setattr(job_service.time, "time", lambda: current_time[0])

    first = client.post("/api/v1/node/jobs/claim", headers=node_headers).json()
    current_time[0] += JOB_LEASE_SECONDS + 1
    second = client.post("/api/v1/node/jobs/claim", headers=node_headers).json()
    current_time[0] += JOB_LEASE_SECONDS + 1
    third = client.post("/api/v1/node/jobs/claim", headers=node_headers).json()
    current_time[0] += JOB_LEASE_SECONDS + 1
    exhausted = client.post("/api/v1/node/jobs/claim", headers=node_headers)
    jobs = client.get("/api/v1/jobs", headers=user_headers(token)).json()

    assert first["lease_token"] != second["lease_token"] != third["lease_token"]
    assert first["job_id"] == second["job_id"] == third["job_id"]
    assert exhausted.status_code == 204
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["failure_code"] == "lease_expired"
    assert jobs[0]["attempt_count"] == 3


def test_expired_lease_result_is_rejected(client: TestClient, monkeypatch):
    register(client, "alice@example.com")
    token = login(client, "alice@example.com")
    node = create_node(client, token)
    create_job(client, token, values=[4, 5])
    node_headers = user_headers(node["node_token"])
    now = int(job_service.time.time())
    current_time = [now]
    monkeypatch.setattr(job_service.time, "time", lambda: current_time[0])
    claim = client.post("/api/v1/node/jobs/claim", headers=node_headers).json()
    current_time[0] += JOB_LEASE_SECONDS + 1

    expired = client.post(
        f"/api/v1/node/jobs/{claim['job_id']}/result",
        headers=node_headers,
        json={
            "lease_token": claim["lease_token"],
            "result": {"sum": 9},
        },
    )

    assert expired.status_code == 409


def test_node_agent_runs_one_job_using_typed_http_protocol(client: TestClient):
    register(client, "alice@example.com")
    token = login(client, "alice@example.com")
    node = create_node(client, token)
    create_job(client, token, values=[6, 8])

    def forward(request: httpx.Request) -> httpx.Response:
        response = client.request(
            request.method,
            request.url.path,
            headers={"Authorization": request.headers["Authorization"]},
            content=request.content,
        )
        return httpx.Response(
            response.status_code,
            headers=dict(response.headers),
            content=response.content,
            request=request,
        )

    agent = NodeAgent(
        server_url="http://ai-home.test",
        node_token=node["node_token"],
        transport=httpx.MockTransport(forward),
    )
    try:
        did_work = agent.run_once()
        no_work = agent.run_once()
    finally:
        agent.close()

    job = client.get("/api/v1/jobs", headers=user_headers(token)).json()[0]
    assert did_work is True
    assert no_work is False
    assert job["status"] == "completed"
    assert job["result"] == {"sum": 14}


def test_node_onboarding_reports_detected_hardware(
    client: TestClient,
    monkeypatch,
):
    register(client, "alice@example.com")
    token = login(client, "alice@example.com")
    monkeypatch.setattr(
        "ai_home.node_worker.detect_hardware",
        lambda: HardwareReport.model_validate(hardware_report()),
    )

    def forward(request: httpx.Request) -> httpx.Response:
        response = client.request(
            request.method,
            request.url.path,
            headers={"Authorization": request.headers["Authorization"]},
            content=request.content,
        )
        return httpx.Response(
            response.status_code,
            headers=dict(response.headers),
            content=response.content,
            request=request,
        )

    registration = register_node_with_server(
        server_url="http://ai-home.test",
        user_token=token,
        name="desktop-node",
        transport=httpx.MockTransport(forward),
    )

    assert registration.node_id
    assert registration.node_token
    assert registration.supported_job_types == ["math.sum.v1"]


def test_account_deletion_removes_owned_nodes_and_jobs(client: TestClient):
    register(client, "alice@example.com")
    token = login(client, "alice@example.com")
    create_node(client, token)
    create_job(client, token)

    deleted = client.request(
        "DELETE",
        "/api/v1/users/me",
        headers=user_headers(token),
        json={"password": "a-strong-password-123"},
    )

    assert deleted.status_code == 204
    with client.app.state.session_factory() as session:
        assert session.scalar(select(User).where(User.email == "alice@example.com")) is None
        assert session.scalar(select(ComputeNode)) is None


def test_math_job_rejects_boolean_and_requires_bounded_input(client: TestClient):
    register(client, "alice@example.com")
    token = login(client, "alice@example.com")

    boolean = create_job(client, token, values=[True])
    excessive = create_job(client, token, values=[10**13])

    assert boolean.status_code == 422
    assert excessive.status_code == 422


def test_nvidia_smi_parser_keeps_gpu_memory_per_device():
    devices = _parse_nvidia_smi(
        '0, NVIDIA GeForce RTX 4070 SUPER, 12282\n'
        '1, NVIDIA GeForce GTX 1050 Ti, 4096\n'
    )

    assert devices == [
        {
            "device_index": 0,
            "name": "NVIDIA GeForce RTX 4070 SUPER",
            "memory_total_bytes": 12282 * 1024**2,
        },
        {
            "device_index": 1,
            "name": "NVIDIA GeForce GTX 1050 Ti",
            "memory_total_bytes": 4096 * 1024**2,
        },
    ]


def test_hardware_report_rejects_duplicate_gpu_indices():
    duplicate = hardware_report()
    duplicate["gpus"][1]["device_index"] = 0

    with pytest.raises(ValueError, match="indices must be unique"):
        HardwareReport.model_validate(duplicate)
