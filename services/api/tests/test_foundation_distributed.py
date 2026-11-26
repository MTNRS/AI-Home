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
    claim = client.post("/api/v1/node/jobs/claim", headers=node_headers).json()
    result = execute_gradient(claim["payload"])
    submit = f"/api/v1/node/jobs/{claim['job_id']}/result"
    assert client.post(submit, headers=node_headers, json={"lease_token": claim["lease_token"],
        "result": {**result, "token_count": 0}}).status_code == 422
    assert client.post(submit, headers=node_headers, json={"lease_token": claim["lease_token"],
        "result": result}).json()["status"] == "completed"
