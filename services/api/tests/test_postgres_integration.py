import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, delete, inspect, select
from sqlalchemy.orm import Session

from fastapi.testclient import TestClient

from ai_home.api import create_app
from ai_home.config import Settings
from ai_home.credits import CreditLedger, InsufficientCredits
from ai_home.database import Base
from ai_home.jobs import claim_job, create_math_sum_job
from ai_home.models import ComputeNode, ComputeJob, User

DATABASE_URL = os.getenv("AI_HOME_TEST_POSTGRES_URL")


@pytest.fixture(scope="module")
def postgres_engine():
    if not DATABASE_URL:
        pytest.skip("Set AI_HOME_TEST_POSTGRES_URL to run PostgreSQL integration tests.")

    with pytest.MonkeyPatch.context() as patch:
        # Alembic reads AI_HOME_DATABASE_URL, not the test URL. Bind both explicitly.
        patch.setenv("AI_HOME_DATABASE_URL", DATABASE_URL)
        patch.setenv("AI_HOME_ENVIRONMENT", "test")
        config = Config("alembic.ini")
        command.upgrade(config, "head")
        engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=5)
        try:
            yield engine
        finally:
            engine.dispose()


def test_postgres_migrations_match_orm(postgres_engine):
    command.check(Config("alembic.ini"))
    assert {
        "users",
        "auth_sessions",
        "credit_accounts",
        "credit_transactions",
        "credit_postings",
        "compute_nodes",
        "compute_jobs",
    } <= set(inspect(postgres_engine).get_table_names())


def test_postgres_identity_api_round_trip_and_account_deletion(postgres_engine):
    app = create_app(
        Settings(environment="test", database_url=DATABASE_URL),
        engine=postgres_engine,
    )
    email = f"{uuid4()}@example.com"
    with TestClient(app) as client:
        registered = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "a-strong-password-123"},
        )
        login = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": "a-strong-password-123"},
        )
        assert registered.status_code == 201
        assert login.status_code == 200
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        profile = client.get("/api/v1/users/me", headers=headers)
        deleted = client.request(
            "DELETE",
            "/api/v1/users/me",
            headers=headers,
            json={"password": "a-strong-password-123"},
        )
        expired_session = client.get("/api/v1/users/me", headers=headers)

    assert profile.status_code == 200
    assert profile.json()["email"] == email
    assert deleted.status_code == 204
    assert expired_session.status_code == 401
    with Session(postgres_engine) as session:
        assert session.scalar(select(User).where(User.email == email)) is None


def test_postgres_scheduler_concurrent_claim_is_single_assignment(postgres_engine):
    owner_id = str(uuid4())
    node_ids = [str(uuid4()), str(uuid4())]
    with Session(postgres_engine) as session, session.begin():
        session.add(
            User(
                user_id=owner_id,
                email=f"{owner_id}@example.com",
                password_hash="test-only",
            )
        )
        session.flush()
        nodes = [
            ComputeNode(
                node_id=node_id,
                owner_user_id=owner_id,
                name=f"node-{index}",
                credential_hash=f"{uuid4().hex}{uuid4().hex}",
                hardware={
                    "platform": "CI",
                    "cpu_model": "PostgreSQL runner",
                    "logical_cores": 2,
                    "ram_total_bytes": 1,
                    "gpus": [],
                },
                supported_job_types=["math.sum.v1"],
                last_heartbeat_epoch=4_000_000_000,
            )
            for index, node_id in enumerate(node_ids)
        ]
        session.add_all(nodes)
        job = create_math_sum_job(
            session,
            owner_user_id=owner_id,
            idempotency_key=str(uuid4()),
            values=[12, 30],
            allow_public_node_processing=True,
        )
        job_id = job.job_id

    barrier = Barrier(2)

    def claim(node_id: str) -> str | None:
        with Session(postgres_engine) as session, session.begin():
            node = session.get(ComputeNode, node_id)
            assert node is not None
            barrier.wait(timeout=10)
            claimed = claim_job(session, node)
            return claimed.job.job_id if claimed else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(claim, node_ids))

    assert claims.count(job_id) == 1
    assert claims.count(None) == 1
    with Session(postgres_engine) as session:
        stored_job = session.get(ComputeJob, job_id)
        assert stored_job is not None
        assert stored_job.status == "leased"
        assert stored_job.assigned_node_id in node_ids

    with Session(postgres_engine) as session, session.begin():
        session.execute(delete(ComputeJob).where(ComputeJob.owner_user_id == owner_id))
        session.execute(delete(ComputeNode).where(ComputeNode.owner_user_id == owner_id))
        session.execute(delete(User).where(User.user_id == owner_id))


def test_postgres_ledger_prevents_concurrent_overdraft(postgres_engine):
    suffix = uuid4().hex
    system_account = f"ci:system:{suffix}"
    user_account = f"ci:user:{suffix}"
    reserve_account = f"ci:reserve:{suffix}"
    ledger = CreditLedger(postgres_engine)
    ledger.create_account(system_account, "system", allow_negative=True)
    ledger.create_account(user_account, "user_available")
    ledger.create_account(reserve_account, "user_reserved")
    ledger.post(
        idempotency_key=f"ci:issue:{suffix}",
        operation="issue",
        postings={system_account: -100, user_account: 100},
    )

    barrier = Barrier(2)

    def reserve(key: str) -> str:
        barrier.wait(timeout=10)
        try:
            ledger.post(
                idempotency_key=key,
                operation="reserve",
                postings={user_account: -80, reserve_account: 80},
            )
            return "accepted"
        except InsufficientCredits:
            return "insufficient"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve, [f"ci:r1:{suffix}", f"ci:r2:{suffix}"]))

    assert sorted(results) == ["accepted", "insufficient"]
    assert ledger.balance(user_account) == 20
    assert ledger.balance(reserve_account) == 80



def test_postgres_inference_duplicate_settlement_and_expiry(postgres_engine):
    from datetime import datetime, timezone, timedelta
    from ai_home.inference_credits import reserve_inference, resolve_inference, refund_expired_inference
    from ai_home.credits import InvalidPosting
    from ai_home.job_credits import user_account_id
    owner = str(uuid4())
    issuer = "pg:issuer:" + owner
    ledger = CreditLedger(postgres_engine)
    ledger.create_account(issuer, "test", allow_negative=True)
    ledger.create_account(user_account_id(owner), "user")
    ledger.post(idempotency_key="pg:fund:"+owner, operation="test",
                postings={issuer:-20, user_account_id(owner):20})
    barrier = Barrier(2)
    def reserve(_):
        barrier.wait(timeout=10)
        with Session(postgres_engine) as session, session.begin():
            row, created = reserve_inference(session, owner, "same", "a"*64, 7)
            return row.request_id, created
    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(reserve, range(2)))
    assert sum(created for _, created in rows) == 1
    key = rows[0][0]
    def settle(_):
        barrier.wait(timeout=10)
        with Session(postgres_engine) as session, session.begin():
            resolve_inference(session, key, owner)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(settle, range(2)))
    assert ledger.balance(user_account_id(owner)) == 13
    with Session(postgres_engine) as session, session.begin():
        stale, _ = reserve_inference(session, owner, "expired", "b"*64, 5)
        stale.expires_at = datetime.now(timezone.utc)-timedelta(seconds=1)
        stale_key = stale.request_id
    with pytest.raises(InvalidPosting):
        with Session(postgres_engine) as session, session.begin():
            resolve_inference(session, stale_key, owner)
    with Session(postgres_engine) as session, session.begin():
        assert refund_expired_inference(session, owner) == 1
    assert ledger.balance(user_account_id(owner)) == 13

def test_postgres_scheduler_measurement_json_predicate(postgres_engine):
    import time
    from ai_home.jobs import create_gradient_job, _measurement_eligibility
    from test_foundation_distributed import setup_shards
    _, _, _, payloads = setup_shards()
    owner = str(uuid4())
    now = int(time.time())
    with Session(postgres_engine) as session:
        try:
            session.add(User(user_id=owner, email=f'{owner}@example.com', password_hash='test-only'))
            session.flush()
            jobs = [create_gradient_job(session, owner_user_id=owner, idempotency_key=str(uuid4()),
                payload=payloads[0], allow_public_node_processing=True, scheduling=constraint)
                for constraint in (None, {'min_gradient_tokens_per_second':1.0},
                                   {'min_gradient_tokens_per_second':100.0},
                                   {'min_transfer_download_bytes_per_second':1.0,
                                    'min_transfer_upload_bytes_per_second':1.0,
                                    'max_transfer_http_round_trip_seconds':1.0})]
            sample = {'format':'ai-home-worker-service-measurement-v1', 'job_type':'foundation.gradient.v1',
                      'device':'cpu', 'heartbeat_service_seconds':0.01, 'claim_service_seconds':0.02,
                      'execution_seconds':1.0, 'submit_service_seconds':0.03,
                      'claim_response_bytes':1000, 'submit_request_bytes':1000}
            node = ComputeNode(hardware={'resources_epoch':now,'resources':{'foundation_device':'cpu'},
                'service_measurement':{'completed_epoch':now,'sample':sample,
                'profile_sha256':jobs[1].scheduling['profile_sha256'], 'gradient_tokens_per_second':10.0,
                'claim_service_bytes_per_second':50000.0,'submit_service_bytes_per_second':30000.0}})
            query=select(ComputeJob.job_id).where(ComputeJob.owner_user_id==owner)
            assert set(session.scalars(query.where(_measurement_eligibility(node, now)))) == {jobs[0].job_id,jobs[1].job_id}
            from ai_home.network_probe import TransferMeasurement
            node.hardware['transfer_measurement'] = TransferMeasurement(
                format='ai-home-http-transfer-probe-v1', payload_bytes=65536,
                empty_round_trip_seconds=[.01,.02,.03], download_seconds=.1, upload_seconds=.2).summary(now)
            assert set(session.scalars(query.where(_measurement_eligibility(node, now)))) == {jobs[0].job_id,jobs[1].job_id,jobs[3].job_id}
            assert set(session.scalars(query.where(_measurement_eligibility(node, now+121)))) == {jobs[0].job_id}
        finally:
            session.rollback()

def test_postgres_job_summary_keyset_pagination(postgres_engine):
    from datetime import datetime, timezone
    from sqlalchemy import update
    email = f'{uuid4()}@example.com'
    owner = None
    app = create_app(Settings(environment='test', database_url=DATABASE_URL), engine=postgres_engine)
    try:
        with TestClient(app) as client:
            credentials = {'email':email,'password':'a-strong-password-123'}
            registration=client.post('/api/v1/auth/register',json=credentials)
            registration.raise_for_status()
            owner=registration.json()['user_id']
            login=client.post('/api/v1/auth/login',json=credentials)
            login.raise_for_status()
            headers={'Authorization':'Bearer '+login.json()['access_token']}
            ids=[]
            for i in range(5):
                job=client.post('/api/v1/jobs',headers=headers,json={'values':[1,2],
                    'idempotency_key':str(i),'allow_public_node_processing':True})
                job.raise_for_status(); ids.append(job.json()['job_id'])
            with Session(postgres_engine) as session, session.begin():
                session.execute(update(ComputeJob).where(ComputeJob.owner_user_id==owner).values(
                    created_at=datetime(2026,11,26,12,tzinfo=timezone.utc)))
            first=client.get('/api/v1/jobs/summary',headers=headers,params={'limit':2}).json()
            last=first[-1]
            second=client.get('/api/v1/jobs/summary',headers=headers,params={'limit':3,
                'before_created_at':last['created_at'],'before_job_id':last['job_id']}).json()
            observed=[row['job_id'] for row in first+second]
            assert observed == sorted(ids,reverse=True)
    finally:
        if owner:
            with Session(postgres_engine) as session, session.begin():
                session.execute(delete(User).where(User.user_id==owner))
