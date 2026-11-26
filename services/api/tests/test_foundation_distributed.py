import copy
from dataclasses import asdict
import json

import pytest

torch = pytest.importorskip("torch")

from ai_home.foundation.data import canonical_hash, prepare_records
from ai_home.foundation.distributed import (GradientPayload, aggregate_gradients, encode_tensors,
    execute_gradient, verified_gradient_result)
from ai_home.foundation.model import FoundationConfig, FoundationModel
from ai_home.foundation.tokenizer import ByteBPE
from ai_home.foundation.training import _sequences
from ai_home.node_worker import execute_job
from test_nodes_and_jobs import client, register, login, user_headers, hardware_report


def setup_shards():
    torch.set_num_threads(1)
    records = [{"source": "authored neural shard fixture", "url": "local:tests", "license": "Apache-2.0",
        "date": "2026-11-26", "language": "es", "training_allowed": True,
        "privacy_reviewed": True, "visibility": "public", "text": f"Documento {i}: datos autorizados."}
        for i in range(4)]
    tokenizer = ByteBPE.train([r["text"] for r in records], vocab_size=280)
    config = FoundationConfig(vocab_size=tokenizer.vocab_size, width=16, layers=1, heads=2,
                              kv_heads=1, hidden=32, context=16)
    torch.manual_seed(12)
    model = FoundationModel(config)
    parameters = encode_tensors(dict(model.named_parameters()))
    payloads = [{"shard": {"records": records[i:i+2], "tokenizer": tokenizer.to_dict()},
        "config": asdict(config), "parameters": parameters, "torch_version": str(torch.__version__),
        "weights_public_and_redistributable": True} for i in (0, 2)]
    return model, tokenizer, records, payloads


def test_two_gradients_equal_monolithic_update_and_reject_replay():
    model, tokenizer, records, payloads = setup_shards()
    reference = FoundationModel(model.config)
    reference.load_state_dict(model.state_dict())
    outputs = [execute_gradient(p) for p in payloads]
    assert all(verified_gradient_result(p, r) for p, r in zip(payloads, outputs))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stats = aggregate_gradients(model, tokenizer, outputs, optimizer)
    x, y = _sequences(prepare_records(records), tokenizer, model.config.context)
    loss = torch.nn.functional.cross_entropy(reference(x)[0].flatten(0, 1), y.flatten())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    torch.optim.AdamW(reference.parameters(), lr=1e-3).step()
    assert stats["tokens"] == int((y != -100).sum())
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="Stale"):
        aggregate_gradients(model, tokenizer, outputs, optimizer)
    with pytest.raises(ValueError, match="distinct"):
        aggregate_gradients(model, tokenizer, [outputs[0], outputs[0]], optimizer)


def test_neural_task_rejects_tampering_and_requires_opt_in():
    _, _, _, payloads = setup_shards()
    with pytest.raises(ValueError, match="opt-in"):
        execute_job("foundation.gradient.v1", payloads[0])
    result = execute_job("foundation.gradient.v1", payloads[0], allow_foundation_training=True)
    artifact = json.loads(result["artifact_json"])
    artifact["mean_loss"] += 1
    tampered = {**result, "artifact_json": json.dumps(artifact), "artifact_sha256": canonical_hash(artifact)}
    assert not verified_gradient_result(payloads[0], tampered)
    for key, value in (("weights_public_and_redistributable", False), ("torch_version", "incompatible")):
        with pytest.raises(ValueError):
            GradientPayload.model_validate({**payloads[0], key: value})
    bad = copy.deepcopy(payloads[0])
    bad["parameters"]["embedding.weight"]["data"] = "not-a-tensor"
    with pytest.raises(ValueError):
        GradientPayload.model_validate(bad)


def test_aggregation_rejects_overlapping_documents():
    model, tokenizer, records, payloads = setup_shards()
    overlapping = copy.deepcopy(payloads[1])
    overlapping["shard"]["records"] = [records[0], records[3]]
    results = [execute_gradient(payloads[0]), execute_gradient(overlapping)]
    with pytest.raises(ValueError, match="Overlapping"):
        aggregate_gradients(model, tokenizer, results, torch.optim.AdamW(model.parameters()))


def test_gradient_jobs_run_through_scheduler_and_verify_results(client):
    _, _, _, payloads = setup_shards()
    register(client, "trainer@example.com")
    token = login(client, "trainer@example.com")
    headers = user_headers(token)
    request = {"payload": payloads[0], "idempotency_key": "gradient-0", "allow_public_node_processing": True}
    route = "/api/v1/jobs/foundation/gradient"
    assert client.post(route, headers=headers, json=request).status_code == 503
    client.app.state.settings.foundation_distributed_training_enabled = True
    assert client.post(route, headers=headers, json=request).status_code == 201
    workers = []
    for version in ("incompatible", str(torch.__version__)):
        response = client.post("/api/v1/nodes", headers=headers, json={"name": version,
            "hardware": hardware_report(), "supported_job_types": ["foundation.gradient.v1"],
            "foundation_torch_version": version})
        assert response.status_code == 201
        workers.append(response.json())
    assert client.post("/api/v1/node/jobs/claim", headers=user_headers(workers[0]["node_token"])).status_code == 204
    node_headers = user_headers(workers[1]["node_token"])
    client.app.state.settings.foundation_distributed_training_enabled = False
    assert client.post("/api/v1/node/jobs/claim", headers=node_headers).status_code == 204
    client.app.state.settings.foundation_distributed_training_enabled = True
    heartbeat = "/api/v1/node/heartbeat"
    assert client.post(heartbeat, headers=node_headers, json={"ram_available_bytes": 2**30,
                       "foundation_device": "cuda"}).status_code == 422
    for snapshot in ({"ram_available_bytes": 1, "foundation_device": "cpu"},
                     {"ram_available_bytes": 2**30, "foundation_device": "cuda", "cuda_free_bytes": 1}):
        assert client.post(heartbeat, headers=node_headers, json=snapshot).status_code == 200
        assert client.post("/api/v1/node/jobs/claim", headers=node_headers).status_code == 204
    healthy = {"ram_available_bytes": 2**30, "foundation_device": "cpu"}
    assert client.post(heartbeat, headers=node_headers, json=healthy).status_code == 200
    from ai_home.models import ComputeNode
    with client.app.state.session_factory() as session, session.begin():
        node = session.get(ComputeNode, workers[1]["node_id"])
        node.hardware = {**node.hardware, "resources_epoch": 0}
    # Claim's liveness heartbeat cannot silently freshen an old resource snapshot.
    assert client.post("/api/v1/node/jobs/claim", headers=node_headers).status_code == 204
    assert client.post(heartbeat, headers=node_headers, json=healthy).status_code == 200
    claim = client.post("/api/v1/node/jobs/claim", headers=node_headers).json()
    result = execute_gradient(claim["payload"])
    submit = f"/api/v1/node/jobs/{claim['job_id']}/result"
    assert client.post(submit, headers=node_headers, json={"lease_token": claim["lease_token"],
        "result": {**result, "token_count": 0}}).status_code == 422
    assert client.post(submit, headers=node_headers, json={"lease_token": claim["lease_token"],
        "result": result}).json()["status"] == "completed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_gradient_worker_is_independently_verified_on_cpu(monkeypatch):
    model, tokenizer, _, payloads = setup_shards()
    seen = []
    original = FoundationModel.forward
    def observed(self, tokens, *args, **kwargs):
        seen.append(tokens.device.type)
        return original(self, tokens, *args, **kwargs)
    monkeypatch.setattr(FoundationModel, "forward", observed)
    result = execute_job("foundation.gradient.v1", payloads[0],
                         allow_foundation_training=True, foundation_device="cuda")
    assert seen and set(seen) == {"cuda"}
    seen.clear()
    assert verified_gradient_result(payloads[0], result)
    assert set(seen) == {"cpu"}
    other = execute_gradient(payloads[1])
    before = model.embedding.weight.detach().clone()
    aggregate_gradients(model, tokenizer, [result, other], torch.optim.AdamW(model.parameters(), lr=.001))
    assert not torch.equal(before, model.embedding.weight)

def test_scheduler_enforces_fresh_comparable_service_constraints(client):
    from ai_home.models import ComputeNode
    _, _, _, payloads = setup_shards()
    register(client, 'constraints@example.com')
    headers = user_headers(login(client, 'constraints@example.com'))
    client.app.state.settings.foundation_distributed_training_enabled = True
    route = '/api/v1/jobs/foundation/gradient'
    def enqueue(key, constraints=None, payload=None):
        body = {'payload': payload or payloads[0], 'idempotency_key': key, 'allow_public_node_processing': True}
        if constraints is not None:
            body['scheduling'] = constraints
        response = client.post(route, headers=headers, json=body)
        assert response.status_code == 201, response.text
        return response.json()
    for i, constraint in enumerate(({'min_gradient_tokens_per_second': 1e8},
            {'min_claim_service_bytes_per_second': 1e9}, {'min_submit_service_bytes_per_second': 1e9},
            {'max_heartbeat_service_seconds': 0.001},
            {'min_transfer_download_bytes_per_second': 1e10},
            {'min_transfer_upload_bytes_per_second': 1e10},
            {'max_transfer_http_round_trip_seconds': .0001})):
        enqueue(f'blocked-{i}', constraint)
    acceptable = {'min_gradient_tokens_per_second': 1.0, 'min_claim_service_bytes_per_second': 1.0,
                  'min_submit_service_bytes_per_second': 1.0, 'max_heartbeat_service_seconds': 1.0,
                  'min_transfer_download_bytes_per_second': 1.0,
                  'min_transfer_upload_bytes_per_second': 1.0, 'max_transfer_http_round_trip_seconds': 1.0}
    eligible = enqueue('eligible', acceptable)
    assert eligible['scheduling']['profile_sha256']
    assert enqueue('eligible', acceptable)['job_id'] == eligible['job_id']
    assert client.post(route, headers=headers, json={'payload': payloads[0], 'idempotency_key': 'eligible',
        'allow_public_node_processing': True, 'scheduling': {'min_gradient_tokens_per_second': 2.0}}).status_code == 409
    assert client.post(route, headers=headers, json={'payload': payloads[0], 'idempotency_key': 'empty',
        'allow_public_node_processing': True, 'scheduling': {}}).status_code == 422
    bootstrap = enqueue('bootstrap')
    node = client.post('/api/v1/nodes', headers=headers, json={'name':'measured',
        'hardware':hardware_report(), 'supported_job_types':['foundation.gradient.v1'],
        'foundation_torch_version':str(torch.__version__)}).json()
    nh = user_headers(node['node_token'])
    resource = {'ram_available_bytes':2**30, 'foundation_device':'cpu'}
    client.post('/api/v1/node/heartbeat', headers=nh, json=resource).raise_for_status()
    claim = client.post('/api/v1/node/jobs/claim', headers=nh).json()
    assert claim['job_id'] == bootstrap['job_id']  # constrained head does not starve bootstrap
    result = execute_gradient(claim['payload'])
    client.post(f"/api/v1/node/jobs/{claim['job_id']}/result", headers=nh,
        json={'lease_token':claim['lease_token'], 'result':result}).raise_for_status()
    sample = {'format':'ai-home-worker-service-measurement-v1', 'job_type':'foundation.gradient.v1',
        'device':'cpu', 'heartbeat_service_seconds':0.01, 'claim_service_seconds':0.02,
        'execution_seconds':1.0, 'submit_service_seconds':0.03,
        'claim_response_bytes':1000, 'submit_request_bytes':1000}
    client.post(f"/api/v1/node/jobs/{claim['job_id']}/measurement", headers=nh, json=sample).raise_for_status()
    client.post('/api/v1/node/heartbeat', headers=nh, json={**resource,'foundation_device':'cuda','cuda_free_bytes':2**30}).raise_for_status()
    assert client.post('/api/v1/node/jobs/claim', headers=nh).status_code == 204
    client.post('/api/v1/node/heartbeat', headers=nh, json=resource).raise_for_status()
    assert client.post('/api/v1/node/jobs/claim', headers=nh).status_code == 204
    transfer = {'format':'ai-home-http-transfer-probe-v1', 'payload_bytes':65536,
        'empty_round_trip_seconds':[.01,.02,.03], 'download_seconds':.1, 'upload_seconds':.2}
    client.post('/api/v1/node/transfer-measurement', headers=nh, json=transfer).raise_for_status()
    claim2 = client.post('/api/v1/node/jobs/claim', headers=nh).json()
    assert claim2['job_id'] == eligible['job_id']
    changed = copy.deepcopy(payloads[0]); changed['config']['context'] = 8
    enqueue('other-profile', acceptable, changed)
    assert client.post('/api/v1/node/jobs/claim', headers=nh).status_code == 204
    enqueue('later', acceptable)
    with client.app.state.session_factory() as session, session.begin():
        stored = session.get(ComputeNode, node['node_id'])
        hardware = copy.deepcopy(stored.hardware)
        hardware['transfer_measurement']['received_epoch'] = 0
        hardware['transfer_measurement']['expires_epoch'] = 120
        stored.hardware = hardware
    assert client.post('/api/v1/node/jobs/claim', headers=nh).status_code == 204
    with client.app.state.session_factory() as session, session.begin():
        stored = session.get(ComputeNode, node['node_id'])
        hardware = copy.deepcopy(stored.hardware)
        hardware['service_measurement']['completed_epoch'] = 0
        stored.hardware = hardware
    assert client.post('/api/v1/node/jobs/claim', headers=nh).status_code == 204
