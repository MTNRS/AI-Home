import copy
import json
import pytest
import torch
from ai_home.foundation.adapters import attach_adapters
from ai_home.foundation.distributed import encode_tensors
from ai_home.foundation.adapter_tasks import execute_adapter_gradient, verified_adapter_gradient
from test_foundation_distributed import setup_shards
from test_nodes_and_jobs import client, register, login, user_headers, hardware_report


def test_malformed_adapter_artifacts_rejected_before_neural_recomputation(monkeypatch):
    import ai_home.foundation.adapter_tasks as tasks
    from ai_home.foundation.data import canonical_hash
    def unexpected_execution(*args, **kwargs):
        pytest.fail("Malformed artifact triggered neural recomputation")
    monkeypatch.setattr(tasks, "execute_adapter_gradient", unexpected_execution)
    artifact = {"format": "ai-home-adapter-gradient-shard-v1", "tokens": 2,
        "mean_loss": 1., "base_adapter_sha256": "0" * 64,
        "dataset_sha256": "0" * 64, "document_hashes": [], "gradients": {}}
    for changes in ({"tokens": True}, {"tokens": 2.0}, {"mean_loss": True},
                    {"mean_loss": float("inf")}, {"mean_loss": float("nan")}):
        changed = {**artifact, **changes}
        assert not tasks.verified_adapter_gradient({}, {"token_count": 2,
            "artifact_json": json.dumps(changed), "artifact_sha256": canonical_hash(changed)})
    for value in (None, [], 7, "[" * 2000 + "0" + "]" * 2000):
        assert not tasks.verified_adapter_gradient({}, {"token_count": 2,
            "artifact_json": value, "artifact_sha256": "0" * 64})


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_adapter_gradient_and_independent_verification(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    base, _, _, payloads = setup_shards()
    model = attach_adapters(base, rank=2, alpha=2.)
    factors = encode_tensors({k: v for k,v in model.named_parameters() if v.requires_grad})
    payload = dict(base=payloads[0], rank=2, alpha=2., factors=factors, factors_public_and_redistributable=True)
    result = execute_adapter_gradient(payload, device=device)
    assert verified_adapter_gradient(payload, result)
    artifact = json.loads(result["artifact_json"])
    assert len(artifact["gradients"]) == 4 and result["token_count"] > 0
    tampered = copy.deepcopy(result)
    tampered["token_count"] += 1
    assert not verified_adapter_gradient(payload, tampered)
    payload["factors_public_and_redistributable"] = False
    with pytest.raises(ValueError, match="Restricted"):
        execute_adapter_gradient(payload, device=device)


def test_adapter_aggregation_matches_full_batch_and_rejects_invalid_shards():
    from ai_home.foundation.adapter_tasks import aggregate_adapter_gradients
    from ai_home.foundation.data import prepare_records
    from ai_home.foundation.training import _sequences
    base, tokenizer, records, bases = setup_shards()
    model = attach_adapters(base, rank=2, alpha=2.)
    reference = copy.deepcopy(model)
    factors = {k:v for k,v in model.named_parameters() if v.requires_grad}
    payloads = [dict(base=p, rank=2, alpha=2., factors=encode_tensors(factors),
                     factors_public_and_redistributable=True) for p in bases]
    results = [execute_adapter_gradient(p) for p in payloads]
    optimizer = torch.optim.AdamW(list(factors.values()), lr=.001)
    initial = {k:v.clone() for k,v in model.state_dict().items()}
    with pytest.raises(ValueError, match="Overlapping"):
        aggregate_adapter_gradients(model, tokenizer, [payloads[0]]*2, [results[0]]*2, optimizer)
    altered = copy.deepcopy(results)
    altered[1]["token_count"] += 1
    with pytest.raises(ValueError, match="Unverified"):
        aggregate_adapter_gradients(model, tokenizer, payloads, altered, optimizer)
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in initial.items())
    assert not optimizer.state
    stats = aggregate_adapter_gradients(model, tokenizer, payloads, results, optimizer)
    x,y = _sequences(prepare_records(records), tokenizer, model.config.context)
    loss = torch.nn.functional.cross_entropy(reference(x)[0].flatten(0,1), y.flatten())
    loss.backward()
    trainable = [p for p in reference.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable, 1.)
    torch.optim.AdamW(trainable, lr=.001).step()
    assert stats["tokens"] == int((y != -100).sum())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="Stale"):
        aggregate_adapter_gradients(model, tokenizer, payloads, results, optimizer)


def test_worker_adapter_permission_is_separate_and_default_off():
    from ai_home.node_worker import execute_job
    base, _, _, payloads = setup_shards()
    model = attach_adapters(base, rank=2, alpha=2.)
    payload = dict(base=payloads[0], rank=2, alpha=2.,
        factors=encode_tensors({k:v for k,v in model.named_parameters() if v.requires_grad}),
        factors_public_and_redistributable=True)
    for options in ({}, {"allow_foundation_training": True}, {"allow_foundation_evaluation": True},
                    {"allow_foundation_adapters": 1}):
        with pytest.raises(ValueError, match="separate explicit"):
            execute_job("foundation.adapter-gradient.v1", payload, **options)
    result = execute_job("foundation.adapter-gradient.v1", payload, allow_foundation_adapters=True)
    assert verified_adapter_gradient(payload, result)


def test_adapter_capability_negotiation_requires_runtime_and_explicit_optin():
    import httpx
    from ai_home.node_worker import NodeAgent
    from ai_home.schemas import NodeCapabilities
    with pytest.raises(ValueError):
        NodeCapabilities(supported_job_types=["foundation.adapter-gradient.v1"])
    sent = []
    def respond(request):
        data = json.loads(request.content)
        NodeCapabilities.model_validate(data)
        sent.append(data)
        return httpx.Response(200, json=data)
    for enabled in (False, True):
        agent = NodeAgent(server_url="http://localhost:8000", node_token="fixture",
            transport=httpx.MockTransport(respond), allow_foundation_adapters=enabled)
        try:
            agent.configure_capabilities()
        finally:
            agent._client.close()
    assert "foundation.adapter-gradient.v1" not in sent[0]["supported_job_types"]
    assert "foundation.adapter-gradient.v1" in sent[1]["supported_job_types"]
    assert sent[1]["foundation_torch_version"] == str(torch.__version__)


def test_adapter_api_scheduler_and_verified_completion(client):
    from ai_home.node_worker import execute_job
    base, _, _, bases = setup_shards()
    model = attach_adapters(base, rank=2, alpha=2.)
    payload = dict(base=bases[0], rank=2, alpha=2.,
        factors=encode_tensors({k:v for k,v in model.named_parameters() if v.requires_grad}),
        factors_public_and_redistributable=True)
    register(client, "adapter-owner@example.com")
    headers = user_headers(login(client, "adapter-owner@example.com"))
    request = dict(payload=payload, idempotency_key="adapter-one", allow_public_node_processing=True)
    route = "/api/v1/jobs/foundation/adapter-gradient"
    assert client.post(route, headers=headers, json=request).status_code == 503
    client.app.state.settings.foundation_distributed_adapters_enabled = True
    assert client.post(route, headers=headers, json={**request, "allow_public_node_processing": False}).status_code == 422
    response = client.post(route, headers=headers, json=request)
    assert response.status_code == 201, response.text
    assert client.post(route, headers=headers, json=request).json()["job_id"] == response.json()["job_id"]
    for version in ("incompatible", str(torch.__version__)):
        response = client.post("/api/v1/nodes", headers=headers, json={"name": version,
            "hardware": hardware_report(), "supported_job_types": ["foundation.adapter-gradient.v1"],
            "foundation_torch_version": version})
        assert response.status_code == 201, response.text
        node_headers = user_headers(response.json()["node_token"])
        client.post("/api/v1/node/heartbeat", headers=node_headers,
                    json={"ram_available_bytes": 2**30, "foundation_device": "cpu"})
        claim = client.post("/api/v1/node/jobs/claim", headers=node_headers)
        if version == "incompatible":
            assert claim.status_code == 204
            continue
        assert claim.status_code == 200, claim.text
        job = claim.json()
        result = execute_job(job["job_type"], job["payload"], allow_foundation_adapters=True)
        submit = f"/api/v1/node/jobs/{job['job_id']}/result"
        assert client.post(submit, headers=node_headers, json={"lease_token": job["lease_token"],
            "result": {**result, "token_count": 0}}).status_code == 422
        completed = client.post(submit, headers=node_headers,
            json={"lease_token": job["lease_token"], "result": result})
        assert completed.status_code == 200, completed.text
        assert completed.json()["status"] == "completed"


def test_funded_adapter_verification_settlement_and_refund(client):
    from test_job_credits import funded_users
    from ai_home.node_worker import execute_job
    from ai_home.credits import Account
    from sqlalchemy import select, func
    _, _, owner, volunteer, node = funded_users(client)
    client.app.state.settings.foundation_distributed_adapters_enabled = True
    response = client.put("/api/v1/node/capabilities", headers=node, json={
        "supported_job_types": ["foundation.adapter-gradient.v1"],
        "foundation_torch_version": str(torch.__version__)})
    assert response.status_code == 200
    client.post("/api/v1/node/heartbeat", headers=node,
                json={"ram_available_bytes": 2**30, "foundation_device": "cpu"})
    base, _, _, bases = setup_shards()
    adapted = attach_adapters(base, rank=2, alpha=2.)
    payload = dict(base=bases[0], rank=2, alpha=2.,
        factors=encode_tensors({k:v for k,v in adapted.named_parameters() if v.requires_grad}),
        factors_public_and_redistributable=True)
    route = "/api/v1/jobs/foundation/adapter-gradient"
    request = dict(payload=payload, idempotency_key="funded-adapter", allow_public_node_processing=True, reward_credits=9)
    assert client.post(route, headers=volunteer, json=request).status_code == 402
    created = client.post(route, headers=owner, json=request)
    assert created.status_code == 201, created.text
    assert client.post(route, headers=owner, json=request).json()["job_id"] == created.json()["job_id"]
    assert client.get("/api/v1/credits", headers=owner).json() == {"available": 91, "reserved": 9, "earned": 0}
    claim = client.post("/api/v1/node/jobs/claim", headers=node).json()
    result = execute_job(claim["job_type"], claim["payload"], allow_foundation_adapters=True)
    submit = f"/api/v1/node/jobs/{claim['job_id']}/result"
    assert client.post(submit, headers=node, json={"lease_token": claim["lease_token"],
        "result": {**result, "token_count": 0}}).status_code == 422
    assert client.get("/api/v1/credits", headers=volunteer).json()["available"] == 0
    # Artifact JSON is a string inside the outer request: its parser must reject
    # excessive nesting without losing the lease or settling the reservation.
    for malformed in ('[' * 2000 + '0' + ']' * 2000, 'null', '[]', '{'):
        rejected = client.post(submit, headers=node, json={"lease_token": claim["lease_token"],
            "result": {**result, "artifact_json": malformed}})
        assert rejected.status_code == 422, rejected.text
        assert client.get("/api/v1/credits", headers=volunteer).json()["available"] == 0
        assert client.get("/api/v1/credits", headers=owner).json()["reserved"] == 9
    body = {"lease_token": claim["lease_token"], "result": result}
    assert client.post(submit, headers=node, json=body).status_code == 200
    assert client.post(submit, headers=node, json=body).status_code == 404
    assert client.get("/api/v1/credits", headers=volunteer).json() == {"available": 9, "reserved": 0, "earned": 9}
    second = client.post(route, headers=owner, json={**request, "idempotency_key": "refund-adapter"}).json()
    cancel = f"/api/v1/jobs/{second['job_id']}/cancel"
    assert client.post(cancel, headers=volunteer).status_code == 404
    assert client.post(cancel, headers=owner).status_code == 200
    assert client.post(cancel, headers=owner).status_code == 200
    assert client.get("/api/v1/credits", headers=owner).json() == {"available": 91, "reserved": 0, "earned": 0}
    with client.app.state.session_factory() as session:
        assert session.scalar(select(func.sum(Account.balance))) == 0
