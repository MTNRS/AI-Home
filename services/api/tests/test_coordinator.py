from dataclasses import asdict
import httpx
import pytest
torch = pytest.importorskip("torch")

from ai_home.foundation.checkpoint import FORMAT, atomic_save
from ai_home.foundation.coordinator import advance, create_plan
from ai_home.foundation.distributed import aggregate_gradients, execute_gradient, encode_tensors
from ai_home.credits import CreditLedger
from ai_home.job_credits import user_account_id
from test_foundation_distributed import setup_shards
from test_nodes_and_jobs import client, register, login, user_headers, hardware_report


def test_restart_reuses_jobs_and_credit_reservations(client, tmp_path, monkeypatch):
    client.base_url = httpx.URL("http://127.0.0.1")
    owner = register(client, "coordinator@example.com")
    client.headers.update(user_headers(login(client, "coordinator@example.com")))
    client.app.state.settings.foundation_distributed_training_enabled = True
    ledger = CreditLedger(client.app.state.engine)
    account = user_account_id(owner["user_id"])
    ledger.create_account("fixture-issuance", "test", allow_negative=True)
    ledger.create_account(account, "user")
    ledger.post(idempotency_key="fund-round", operation="test", postings={"fixture-issuance": -20, account: 20})
    model, tokenizer, _, payloads = setup_shards()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    aggregate_gradients(model, tokenizer, [execute_gradient(p) for p in payloads], optimizer)
    parent = tmp_path / "parent.pt"
    atomic_save({"format": FORMAT, "config": asdict(model.config), "tokenizer": tokenizer.to_dict(),
                 "tokenizer_sha256": tokenizer.fingerprint, "model_state": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "step": 1, "data_visibility": "public",
                 "training_mode": "verified-gradient-rounds-v1"}, parent)
    for payload in payloads:
        payload["parameters"] = encode_tensors(dict(model.named_parameters()))
    plan, output = tmp_path / "plan.json", tmp_path / "next.pt"
    create_plan(parent, payloads, plan, server=str(client.base_url), owner_user_id=owner["user_id"],
                reward_credits=3, allow_public_node_processing=True)
    post = client.post
    submissions = []
    def lost_reply(url, **kwargs):
        response = post(url, **kwargs)
        submissions.append(response.json()["job_id"])
        if len(submissions) == 2:
            raise httpx.ReadError("simulated lost reply after server commit")
        return response
    with monkeypatch.context() as patch:
        patch.setattr(client, "post", lost_reply)
        with pytest.raises(httpx.ReadError):
            advance(client, plan, parent, output)
    first = advance(client, plan, parent, output)
    assert [job["id"] for job in first["jobs"]] == submissions
    assert first["state"] == "waiting" and ledger.balance(account) == 14
    assert advance(client, plan, parent, output) == first
    assert ledger.balance(account) == 14
    assert len(client.get("/api/v1/jobs").json()) == 2
    node = client.post("/api/v1/nodes", json={"name": "worker", "hardware": hardware_report(),
        "supported_job_types": ["foundation.gradient.v1"], "foundation_torch_version": str(torch.__version__)}).json()
    for _ in range(2):
        headers = user_headers(node["node_token"])
        job = client.post("/api/v1/node/jobs/claim", headers=headers).json()
        result = execute_gradient(job["payload"])
        response = client.post(f"/api/v1/node/jobs/{job['job_id']}/result", headers=headers,
                               json={"lease_token": job["lease_token"], "result": result})
        assert response.status_code == 200
    committed = advance(client, plan, parent, output)
    assert committed["state"] == "committed" and committed["step"] == 2
    before = output.read_bytes()
    assert advance(client, plan, parent, output) == committed
    assert output.read_bytes() == before and len(client.get("/api/v1/jobs").json()) == 2
    # Self-work returns funds to the same account; neither retry mints credits.
    assert ledger.balance(account) == 20
    register(client, "different-owner@example.com")
    client.headers.update(user_headers(login(client, "different-owner@example.com")))
    with pytest.raises(ValueError, match="owner mismatch"):
        advance(client, plan, parent, output)
