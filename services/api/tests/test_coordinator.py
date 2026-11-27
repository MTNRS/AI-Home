from dataclasses import asdict
import httpx
import pytest
torch = pytest.importorskip("torch")

from ai_home.foundation.checkpoint import FORMAT, atomic_save
from ai_home.foundation.coordinator import advance, create_plan, supervise
from ai_home.foundation.distributed import aggregate_gradients, execute_gradient, encode_tensors
from ai_home.credits import CreditLedger
from ai_home.job_credits import user_account_id
from test_foundation_distributed import setup_shards
from test_nodes_and_jobs import client, register, login, user_headers, hardware_report


def test_supervisor_retries_lost_transport_but_not_http_policy_errors(monkeypatch):
    import ai_home.foundation.coordinator as coordinator
    import threading
    clock = [0.]
    monkeypatch.setattr(coordinator.time, "monotonic", lambda: clock[0])
    class Wait:
        def is_set(self):
            return False
        def wait(self, seconds):
            clock[0] += seconds
    outcomes = [httpx.ReadError("lost response"), {"state": "waiting", "jobs": []},
                {"state": "committed", "step": 2}]
    def advance(*args):
        value = outcomes.pop(0)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(coordinator, "advance", advance)
    result = coordinator.supervise(None, None, None, None, wait_seconds=10, poll_seconds=1, cancel=Wait())
    assert result["state"] == "committed" and result["attempts"] == 3 and result["transport_errors"] == 1
    response = httpx.Response(401, request=httpx.Request("GET", "https://example.org"))
    outcomes.append(httpx.HTTPStatusError("unauthorized", request=response.request, response=response))
    with pytest.raises(httpx.HTTPStatusError):
        coordinator.supervise(None, None, None, None, wait_seconds=10, cancel=Wait())
    event = threading.Event(); event.set()
    stopped = coordinator.supervise(None, None, None, None, wait_seconds=10, cancel=event)
    assert stopped["attempts"] == 0 and stopped["state"] == "stopped" and not stopped["jobs_cancelled"]
    monkeypatch.setattr(coordinator, "advance", lambda *args: {"state": "waiting", "jobs": []})
    timed_out = coordinator.supervise(None, None, None, None, wait_seconds=2, poll_seconds=1, cancel=Wait())
    assert timed_out["state"] == "timed_out" and timed_out["attempts"] == 2


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
    observed = supervise(client, plan, parent, output, wait_seconds=10, poll_seconds=1)
    assert observed["state"] == "committed" and observed["attempts"] == 1
    assert output.read_bytes() == before
    # Self-work returns funds to the same account; neither retry mints credits.
    assert ledger.balance(account) == 20
    register(client, "different-owner@example.com")
    client.headers.update(user_headers(login(client, "different-owner@example.com")))
    with pytest.raises(ValueError, match="owner mismatch"):
        advance(client, plan, parent, output)

def test_cli_scheduling_plan_reaches_api_and_rejects_invalid_files(client, tmp_path, monkeypatch):
    import json
    import sys
    from ai_home.foundation.coordinator import main
    from ai_home.foundation.data import canonical_hash
    client.base_url = httpx.URL('http://127.0.0.1')
    owner = register(client, 'cli-planner@example.com')
    client.headers.update(user_headers(login(client, 'cli-planner@example.com')))
    client.app.state.settings.foundation_distributed_training_enabled = True
    model, tokenizer, _, payloads = setup_shards()
    parent = tmp_path / 'parent.pt'
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    aggregate_gradients(model, tokenizer, [execute_gradient(p) for p in payloads], optimizer)
    atomic_save({'format':FORMAT, 'config':asdict(model.config), 'tokenizer':tokenizer.to_dict(),
        'tokenizer_sha256':tokenizer.fingerprint, 'model_state':model.state_dict(),
        'optimizer':optimizer.state_dict(), 'step':1, 'data_visibility':'public',
        'training_mode':'verified-gradient-rounds-v1'}, parent)
    for payload in payloads:
        payload['parameters'] = encode_tensors(dict(model.named_parameters()))
    shards, requirements, plan = (tmp_path / name for name in ('shards.json','requirements.json','plan.json'))
    shards.write_text(json.dumps(payloads), encoding='utf-8')
    constraints = {'min_gradient_tokens_per_second':1.0, 'max_heartbeat_service_seconds':1.0}
    requirements.write_text(json.dumps(constraints), encoding='utf-8-sig')
    monkeypatch.setattr(sys, 'argv', ['coordinator','plan',str(parent),str(shards),str(plan),
        '--server',str(client.base_url),'--owner',owner['user_id'],'--allow-public-node-processing',
        '--scheduling',str(requirements)])
    main()
    data = json.loads(plan.read_text())
    assert data['scheduling'] == constraints
    digest = data.pop('sha256')
    assert canonical_hash(data) == digest
    result = advance(client, plan, parent, tmp_path/'output.pt')
    assert result['state'] == 'waiting'
    jobs = client.get('/api/v1/jobs').json()
    assert len(jobs) == 2
    assert all(job['scheduling']['min_gradient_tokens_per_second'] == 1.0 for job in jobs)
    assert advance(client, plan, parent, tmp_path/'output.pt') == result
    for invalid in ('{}', '{"min_gradient_tokens_per_second": -1}', ' '*65537):
        requirements.write_text(invalid)
        with pytest.raises(ValueError):
            main()
    assert json.loads(plan.read_text())['sha256'] == digest
