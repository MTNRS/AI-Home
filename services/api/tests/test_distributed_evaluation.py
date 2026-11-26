import copy
import json

import pytest

torch = pytest.importorskip("torch")
from ai_home.foundation.data import canonical_hash
from ai_home.foundation.distributed import execute_gradient
from ai_home.foundation.distributed_evaluation import execute_evaluation, verified_evaluation_result
from test_foundation_distributed import setup_shards


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_public_loss_task_matches_gradient_loss_without_mutating_inputs(device):
    _, _, _, shards = setup_shards()
    payload = shards[0]
    before = copy.deepcopy(payload)
    rng = torch.get_rng_state().clone()
    result = execute_evaluation(payload, device=device)
    assert payload == before
    assert torch.equal(rng, torch.get_rng_state())
    assert verified_evaluation_result(payload, result)
    artifact = json.loads(result["artifact_json"])
    gradient = json.loads(execute_gradient(payload, device=device)["artifact_json"])
    assert artifact["tokens"] == gradient["tokens"]
    assert artifact["mean_loss"] == pytest.approx(gradient["mean_loss"], rel=1e-5, abs=1e-6)
    assert artifact["base_sha256"] == gradient["base_sha256"]
    assert "gradients" not in artifact
    for field, replacement in (("mean_loss", artifact["mean_loss"] + 1),
                                ("tokens", True), ("base_sha256", "0" * 64)):
        changed = {**artifact, field: replacement}
        tampered = {**result, "artifact_json": json.dumps(changed), "artifact_sha256": canonical_hash(changed)}
        assert not verified_evaluation_result(payload, tampered)
    assert not verified_evaluation_result(shards[1], result)


def test_evaluation_task_rejects_private_and_nonredistributable_inputs():
    _, _, _, shards = setup_shards()
    for field in ("private", "weights"):
        payload = copy.deepcopy(shards[0])
        if field == "private":
            payload["shard"]["records"][0]["visibility"] = "reviewed-training-only"
        else:
            payload["weights_public_and_redistributable"] = False
        with pytest.raises(ValueError):
            execute_evaluation(payload)


from test_nodes_and_jobs import client, user_headers
from test_job_credits import funded_users


def test_evaluation_api_worker_verification_and_credit_settlement(client):
    import httpx
    from ai_home.node_worker import NodeAgent, execute_job
    _, _, owner_headers, worker_headers, node_headers = funded_users(client)
    _, _, _, shards = setup_shards()
    request = {"payload": shards[0], "allow_public_node_processing": True,
               "idempotency_key": "evaluation-1", "reward_credits": 7}
    route = "/api/v1/jobs/foundation/evaluate"
    assert client.post(route, json=request).status_code == 401
    assert client.post(route, headers=owner_headers, json=request).status_code == 503
    client.app.state.settings.foundation_distributed_evaluation_enabled = True
    assert not client.app.state.settings.foundation_distributed_training_enabled
    assert client.post(route, headers=owner_headers, json={**request, "allow_public_node_processing": False}).status_code == 422
    assert client.post(route, headers=worker_headers, json=request).status_code == 402
    created = client.post(route, headers=owner_headers, json=request)
    assert created.status_code == 201
    assert client.post(route, headers=owner_headers, json=request).json()["job_id"] == created.json()["job_id"]
    assert client.post(route, headers=owner_headers, json={**request, "reward_credits": 8}).status_code == 409
    assert client.post("/api/v1/node/jobs/claim", headers=node_headers).status_code == 204
    capabilities = {"supported_job_types": ["foundation.evaluate.v1"]}
    assert client.put("/api/v1/node/capabilities", headers=node_headers, json=capabilities).status_code == 422
    for version in ("incompatible", str(torch.__version__)):
        assert client.put("/api/v1/node/capabilities", headers=node_headers,
                          json={**capabilities, "foundation_torch_version": version}).status_code == 200
        if version == "incompatible":
            assert client.post("/api/v1/node/jobs/claim", headers=node_headers).status_code == 204
    for ram in (1, 2**30):
        assert client.post("/api/v1/node/heartbeat", headers=node_headers,
                           json={"ram_available_bytes": ram, "foundation_device": "cpu"}).status_code == 200
        if ram == 1:
            assert client.post("/api/v1/node/jobs/claim", headers=node_headers).status_code == 204
    client.app.state.settings.foundation_distributed_evaluation_enabled = False
    assert client.post("/api/v1/node/jobs/claim", headers=node_headers).status_code == 204
    client.app.state.settings.foundation_distributed_evaluation_enabled = True
    claim = client.post("/api/v1/node/jobs/claim", headers=node_headers).json()
    with pytest.raises(ValueError, match="opt-in"):
        execute_job(claim["job_type"], claim["payload"], allow_foundation_training=True)
    result = execute_job(claim["job_type"], claim["payload"], allow_foundation_evaluation=True)
    submit = f"/api/v1/node/jobs/{claim['job_id']}/result"
    assert client.post(submit, headers=node_headers, json={"lease_token": claim["lease_token"],
                       "result": {**result, "token_count": 0}}).status_code == 422
    assert client.get("/api/v1/credits", headers=worker_headers).json()["earned"] == 0
    body = {"lease_token": claim["lease_token"], "result": result}
    assert client.post(submit, headers=node_headers, json=body).status_code == 200
    assert client.post(submit, headers=node_headers, json=body).status_code == 404
    assert client.get("/api/v1/credits", headers=worker_headers).json()["earned"] == 7
    assert client.get("/api/v1/jobs/" + claim["job_id"], headers=worker_headers).status_code == 404
    assert client.post(route, headers=owner_headers, json={**request, "idempotency_key": "evaluation-2"}).status_code == 201
    def transport(request):
        response = client.request(request.method, str(request.url), headers=dict(request.headers), content=request.content)
        return httpx.Response(response.status_code, content=response.content)
    agent = NodeAgent(server_url="http://127.0.0.1", node_token=node_headers["Authorization"].split()[1],
                      allow_foundation_evaluation=True, transport=httpx.MockTransport(transport))
    try:
        agent.configure_capabilities()
        assert agent.run_once()
        assert agent.last_measurement_uploaded
        assert agent.last_run_metrics["job_type"] == "foundation.evaluate.v1"
    finally:
        agent.close()
    assert client.get("/api/v1/credits", headers=worker_headers).json() == {"available": 14, "reserved": 0, "earned": 14}
    assert client.get("/api/v1/credits", headers=owner_headers).json()["reserved"] == 0
