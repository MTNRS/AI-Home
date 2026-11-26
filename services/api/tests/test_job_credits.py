import pytest
from sqlalchemy import func, select

import ai_home.jobs as job_service
from ai_home.credits import Account, CreditLedger, LedgerTransaction, Posting
from ai_home.job_credits import JobFunding, user_account_id
from ai_home.models import ComputeJob, User
from ai_home.node_worker import execute_job
from test_foundation_nodes import payload
from test_nodes_and_jobs import client, register, login, user_headers, hardware_report


def funded_users(client):
    owner = register(client, "owner@example.com")
    worker = register(client, "worker@example.com")
    owner_headers = user_headers(login(client, "owner@example.com"))
    worker_headers = user_headers(login(client, "worker@example.com"))
    ledger = CreditLedger(client.app.state.engine)
    ledger.create_account("test-issuance", "test", allow_negative=True)
    ledger.create_account(user_account_id(owner["user_id"]), "user")
    ledger.post(idempotency_key="test-initial-credits", operation="test_issuance",
                postings={"test-issuance": -100, user_account_id(owner["user_id"]): 100})
    node = client.post("/api/v1/nodes", headers=worker_headers, json={"name": "worker",
        "hardware": hardware_report(), "supported_job_types": ["foundation.tokenize.v1"]}).json()
    return owner, worker, owner_headers, worker_headers, user_headers(node["node_token"])


def submit_job(client, headers, key="paid", reward=7):
    return client.post("/api/v1/jobs/foundation/tokenize", headers=headers, json={
        "payload": payload(), "allow_public_node_processing": True, "idempotency_key": key,
        "reward_credits": reward})


def test_verified_work_settles_once_and_conserves_total_credits(client):
    owner, worker, owner_headers, worker_headers, node_headers = funded_users(client)
    assert client.get("/api/v1/credits").status_code == 401
    assert client.get("/api/v1/credits", headers=worker_headers).json()["available"] == 0
    job = submit_job(client, owner_headers).json()
    assert submit_job(client, owner_headers).json()["job_id"] == job["job_id"]
    assert submit_job(client, owner_headers, reward=8).status_code == 409
    assert client.get("/api/v1/credits", headers=owner_headers).json() == {"available": 93, "reserved": 7, "earned": 0}
    claim = client.post("/api/v1/node/jobs/claim", headers=node_headers).json()
    assert claim["reward_credits"] == 7
    result = execute_job(claim["job_type"], claim["payload"])
    route = f"/api/v1/node/jobs/{claim['job_id']}/result"
    assert client.post(route, headers=node_headers, json={"lease_token": claim["lease_token"],
        "result": {**result, "token_count": 0}}).status_code == 422
    assert client.get("/api/v1/credits", headers=worker_headers).json()["available"] == 0
    submitted = {"lease_token": claim["lease_token"], "result": result}
    assert client.post(route, headers=node_headers, json=submitted).status_code == 200
    assert client.post(route, headers=node_headers, json=submitted).status_code == 404
    assert client.get("/api/v1/credits", headers=worker_headers).json() == {"available": 7, "reserved": 0, "earned": 7}
    assert client.get("/api/v1/credits", headers=owner_headers).json()["reserved"] == 0
    movements = client.get("/api/v1/credits/transactions", headers=worker_headers).json()
    assert [(r["operation"], r["amount"]) for r in movements] == [("job_settle", 7)]
    with client.app.state.session_factory() as session:
        assert session.scalar(select(func.sum(Account.balance))) == 0
        assert session.get(JobFunding, job["job_id"]).status == "settled"
        for tx in session.scalars(select(LedgerTransaction)):
            assert session.scalar(select(func.sum(Posting.amount)).where(Posting.transaction_id == tx.transaction_id)) == 0


def test_insufficient_balance_and_cancellation_refund(client):
    _, _, owner_headers, worker_headers, node_headers = funded_users(client)
    assert submit_job(client, worker_headers).status_code == 402
    assert client.get("/api/v1/jobs", headers=worker_headers).json() == []
    job = submit_job(client, owner_headers).json()
    claim = client.post("/api/v1/node/jobs/claim", headers=node_headers).json()
    route = f"/api/v1/jobs/{job['job_id']}/cancel"
    assert client.post(route, headers=worker_headers).status_code == 404
    assert client.post(route, headers=owner_headers).json()["status"] == "cancelled"
    assert client.post(route, headers=owner_headers).status_code == 200
    assert client.get("/api/v1/credits", headers=owner_headers).json()["available"] == 100
    result = execute_job(claim["job_type"], claim["payload"])
    assert client.post(f"/api/v1/node/jobs/{job['job_id']}/result", headers=node_headers,
        json={"lease_token": claim["lease_token"], "result": result}).status_code == 404


def test_terminal_lease_failure_refunds_reserved_funds(client):
    _, _, owner_headers, _, node_headers = funded_users(client)
    job = submit_job(client, owner_headers).json()
    client.post("/api/v1/node/jobs/claim", headers=node_headers)
    with client.app.state.session_factory() as session:
        stored = session.get(ComputeJob, job["job_id"])
        stored.attempt_count = job_service.MAX_JOB_ATTEMPTS
        stored.lease_expires_epoch = 0
        session.commit()
    assert client.post("/api/v1/node/jobs/claim", headers=node_headers).status_code == 204
    assert client.get("/api/v1/credits", headers=owner_headers).json() == {"available": 100, "reserved": 0, "earned": 0}
    with client.app.state.session_factory() as session:
        assert session.get(ComputeJob, job["job_id"]).status == "failed"
        assert session.get(JobFunding, job["job_id"]).status == "refunded"


def test_settlement_rolls_back_with_job_on_failure(client, monkeypatch):
    _, _, owner_headers, worker_headers, node_headers = funded_users(client)
    submit_job(client, owner_headers)
    claim = client.post("/api/v1/node/jobs/claim", headers=node_headers).json()
    result = execute_job(claim["job_type"], claim["payload"])
    original = job_service.settle_job
    def fail_after_posting(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected failure after ledger write")
    monkeypatch.setattr(job_service, "settle_job", fail_after_posting)
    with pytest.raises(RuntimeError, match="injected"):
        client.post(f"/api/v1/node/jobs/{claim['job_id']}/result", headers=node_headers,
            json={"lease_token": claim["lease_token"], "result": result})
    assert client.get("/api/v1/credits", headers=worker_headers).json()["available"] == 0
    with client.app.state.session_factory() as session:
        assert session.get(ComputeJob, claim["job_id"]).status == "leased"
        assert session.get(JobFunding, claim["job_id"]).status == "reserved"


def test_account_deletion_refunds_pending_jobs_and_preserves_audit(client):
    owner, _, owner_headers, _, _ = funded_users(client)
    job = submit_job(client, owner_headers).json()
    response = client.request("DELETE", "/api/v1/users/me", headers=owner_headers,
                              json={"password": "a-strong-password-123"})
    assert response.status_code == 204
    with client.app.state.session_factory() as session:
        assert session.get(User, owner["user_id"]) is None
        assert session.get(ComputeJob, job["job_id"]) is None
        assert session.get(JobFunding, job["job_id"]).status == "refunded"
        assert session.get(Account, user_account_id(owner["user_id"])).balance == 100


def test_real_neural_gradient_work_earns_funded_credits(client):
    torch = pytest.importorskip("torch")
    from test_foundation_distributed import setup_shards
    _, _, _, payloads = setup_shards()
    _, _, owner_headers, worker_headers, _ = funded_users(client)
    client.app.state.settings.foundation_distributed_training_enabled = True
    node = client.post("/api/v1/nodes", headers=worker_headers, json={"name": "neural-worker",
        "hardware": hardware_report(), "supported_job_types": ["foundation.gradient.v1"],
        "foundation_torch_version": str(torch.__version__)}).json()
    response = client.post("/api/v1/jobs/foundation/gradient", headers=owner_headers, json={
        "payload": payloads[0], "allow_public_node_processing": True,
        "idempotency_key": "paid-neural", "reward_credits": 11})
    assert response.status_code == 201
    node_headers = user_headers(node["node_token"])
    claim = client.post("/api/v1/node/jobs/claim", headers=node_headers).json()
    assert claim["reward_credits"] == 11
    result = execute_job(claim["job_type"], claim["payload"], allow_foundation_training=True)
    response = client.post(f"/api/v1/node/jobs/{claim['job_id']}/result", headers=node_headers,
        json={"lease_token": claim["lease_token"], "result": result})
    assert response.status_code == 200
    assert client.get("/api/v1/credits", headers=worker_headers).json()["earned"] == 11
    assert client.get("/api/v1/credits", headers=owner_headers).json()["available"] == 89


def test_operator_grants_are_audited_and_idempotent_not_work_rewards(client):
    from ai_home.credit_admin import grant_credits
    user = register(client, "member@example.com")
    headers = user_headers(login(client, "member@example.com"))
    ids = []
    for _ in range(2):
        with client.app.state.session_factory() as session, session.begin():
            receipt = grant_credits(session, user_id=user["user_id"], amount=20,
                                    idempotency_key="bootstrap", reason="test allocation")
            ids.append(receipt.transaction_id)
    assert ids[0] == ids[1]
    assert client.get("/api/v1/credits", headers=headers).json() == {"available": 20, "reserved": 0, "earned": 0}
    with client.app.state.session_factory() as session, session.begin():
        with pytest.raises(ValueError, match="does not exist"):
            grant_credits(session, user_id="absent", amount=10, idempotency_key="bad", reason="test")


def test_self_funded_job_cannot_mint_credits_or_inflate_earned_counter(client):
    _, _, owner_headers, _, _ = funded_users(client)
    node = client.post("/api/v1/nodes", headers=owner_headers, json={"name": "self",
        "hardware": hardware_report(), "supported_job_types": ["foundation.tokenize.v1"]}).json()
    submit_job(client, owner_headers)
    node_headers = user_headers(node["node_token"])
    claim = client.post("/api/v1/node/jobs/claim", headers=node_headers).json()
    result = execute_job(claim["job_type"], claim["payload"])
    assert client.post(f"/api/v1/node/jobs/{claim['job_id']}/result", headers=node_headers,
        json={"lease_token": claim["lease_token"], "result": result}).status_code == 200
    assert client.get("/api/v1/credits", headers=owner_headers).json() == {"available": 100, "reserved": 0, "earned": 0}


def test_desktop_credit_client_reads_actual_ledger_without_cross_account_data(client):
    import httpx
    from ai_home.desktop import server_credits
    owner, worker, owner_headers, worker_headers, _ = funded_users(client)
    submit_job(client, owner_headers)
    def bridge(request):
        response = client.get(request.url.path, headers=dict(request.headers))
        return httpx.Response(response.status_code, json=response.json())
    transport = httpx.MockTransport(bridge)
    owner_token = owner_headers["Authorization"].removeprefix("Bearer ")
    worker_token = worker_headers["Authorization"].removeprefix("Bearer ")
    balance, transactions = server_credits("https://example.org", owner_token, transport=transport)
    assert balance == {"available": 93, "reserved": 7, "earned": 0}
    assert any(row["operation"] == "job_reserve" and row["amount"] == -7 for row in transactions)
    worker_balance, worker_transactions = server_credits("https://example.org", worker_token, transport=transport)
    assert worker_balance == {"available": 0, "reserved": 0, "earned": 0}
    assert worker_transactions == []
