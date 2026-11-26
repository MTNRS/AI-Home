import copy
import json

import pytest

from ai_home.foundation.tasks import TokenizePayload, execute_tokenize
from ai_home.foundation.tokenizer import ByteBPE
from ai_home.node_worker import execute_job
from test_nodes_and_jobs import client, register, login, user_headers, hardware_report, create_node


def payload():
    records = [{"source": "authored public fixture", "url": "local:test", "license": "Apache-2.0",
                "date": "2026-11-26", "language": "es", "training_allowed": True,
                "privacy_reviewed": True, "visibility": "public", "text": text}
               for text in ("El modelo aprende con datos revisados.", "Un nodo prepara tokens de texto público.")]
    tokenizer = ByteBPE.train([r["text"] for r in records], vocab_size=280)
    return {"records": records, "tokenizer": tokenizer.to_dict()}


def test_two_capable_nodes_produce_verified_token_shards(client):
    register(client, "owner@example.com")
    owner = login(client, "owner@example.com")
    register(client, "volunteer@example.com")
    volunteer = login(client, "volunteer@example.com")
    legacy = create_node(client, volunteer)
    limited = hardware_report()
    limited["ram_total_bytes"] = 128 * 1024**2
    nodes = []
    for name, hardware in (("low-memory", limited), ("worker-a", hardware_report()), ("worker-b", hardware_report())):
        node = client.post("/api/v1/nodes", headers=user_headers(volunteer), json={
            "name": name, "hardware": hardware, "supported_job_types": ["foundation.tokenize.v1"]})
        assert node.status_code == 201
        nodes.append(node.json())
    request = {"payload": payload(), "idempotency_key": "token-shard-a", "allow_public_node_processing": True}
    route = "/api/v1/jobs/foundation/tokenize"
    assert client.post(route, json=request).status_code == 401
    for consent in (False, 1, "true"):
        assert client.post(route, headers=user_headers(owner), json={
            **request, "allow_public_node_processing": consent}).status_code == 422
    first = client.post(route, headers=user_headers(owner), json=request)
    assert first.status_code == 201
    assert client.post(route, headers=user_headers(owner), json=request).json()["job_id"] == first.json()["job_id"]
    changed = copy.deepcopy(request)
    changed["payload"]["records"][0]["text"] += " Cambio."
    assert client.post(route, headers=user_headers(owner), json=changed).status_code == 409
    changed["idempotency_key"] = "token-shard-b"
    assert client.post(route, headers=user_headers(owner), json=changed).status_code == 201
    for node in (legacy, nodes[0]):
        assert client.post("/api/v1/node/jobs/claim", headers=user_headers(node["node_token"])).status_code == 204
    claims = []
    for node in nodes[1:]:
        response = client.post("/api/v1/node/jobs/claim", headers=user_headers(node["node_token"]))
        assert response.status_code == 200
        claims.append(response.json())
    assert claims[0]["job_id"] != claims[1]["job_id"]
    for node, claim in zip(nodes[1:], claims):
        result = execute_job(claim["job_type"], claim["payload"])
        artifact = json.loads(result["artifact_json"])
        assert result["token_count"] == sum(len(d["tokens"]) for d in artifact["documents"])
        submit = f"/api/v1/node/jobs/{claim['job_id']}/result"
        invalid = {**result, "token_count": result["token_count"] + 1}
        assert client.post(submit, headers=user_headers(node["node_token"]), json={
            "lease_token": claim["lease_token"], "result": invalid}).status_code == 422
        response = client.post(submit, headers=user_headers(node["node_token"]), json={
            "lease_token": claim["lease_token"], "result": result})
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert response.json()["result"] == execute_tokenize(claim["payload"])
    assert client.get("/api/v1/jobs", headers=user_headers(volunteer)).json() == []
    assert len(client.get("/api/v1/jobs", headers=user_headers(owner)).json()) == 2


@pytest.mark.parametrize("change", ["private", "email", "credential_metadata", "budget", "expansion"])
def test_tokenize_rejects_private_or_unbounded_inputs(change):
    value = payload()
    if change == "private":
        value["records"][0]["visibility"] = "reviewed-training-only"
    elif change == "email":
        value["records"][0]["text"] = "Private address person@example.com"
    elif change == "credential_metadata":
        value["records"][0]["source"] = "sk-" + "a" * 48
    elif change == "budget":
        value["records"][0]["text"] = "x" * 20_000
    else:
        # A tiny serialized merge list must not expand into a huge byte string.
        value["tokenizer"]["merges"] = [[65, 65]] + [[i, i] for i in range(262, 280)]
    with pytest.raises(ValueError):
        TokenizePayload.model_validate(value)


def test_unknown_node_task_is_never_executed():
    with pytest.raises(ValueError, match="Unsupported"):
        execute_job("python.exec", {"code": "print('untrusted')"})
