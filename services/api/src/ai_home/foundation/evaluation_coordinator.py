"""Restartable public loss-evaluation plans with checkpoint lineage checks."""
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ai_home.immutable_artifacts import publish_bytes
from ai_home.urls import normalize_base_url
from .checkpoint import load_checkpoint
from .data import canonical_hash, prepare_records
from .distributed import GradientPayload, encode_tensors
from .distributed_evaluation import verified_evaluation_result
from .tasks import PublicRecord


def _payloads(checkpoint, shards):
    if not isinstance(shards, list) or not 1 <= len(shards) <= 8:
        raise ValueError("Require one to eight public evaluation shards")
    model, tokenizer, state = load_checkpoint(checkpoint, max_bytes=10_000_000)
    if state.get("data_visibility") != "public" or "training_document_hashes" not in state:
        raise ValueError("Evaluation requires a public checkpoint with training lineage")
    trained = set(state["training_document_hashes"])
    trained_groups = set(state.get("training_split_groups", []))
    parameters = encode_tensors(dict(model.named_parameters()))
    payloads, seen = [], set()
    for shard in shards:
        records = prepare_records(shard)
        documents = {r["sha256"] for r in records}
        groups = {r["split_group_sha256"] for r in records if "split_group_sha256" in r}
        if groups and "training_split_groups" not in state:
            raise ValueError("Grouped evaluation requires checkpoint group lineage")
        if documents & (trained | seen) or groups & trained_groups:
            raise ValueError("Evaluation contamination or duplicate documents across shards")
        seen.update(documents)
        public = [{key: value for key, value in r.items() if key in PublicRecord.model_fields} for r in records]
        payload = {"shard": {"records": public, "tokenizer": tokenizer.to_dict()},
                   "config": asdict(model.config), "parameters": parameters,
                   "torch_version": str(torch.__version__), "weights_public_and_redistributable": True}
        payloads.append(GradientPayload.model_validate(payload).model_dump())
    return model.checkpoint_sha256, payloads


def create_plan(checkpoint, shards, output, *, server, owner_user_id, reward_credits=0,
                allow_public_node_processing=False, weights_public_and_redistributable=False):
    if allow_public_node_processing is not True or weights_public_and_redistributable is not True:
        raise ValueError("Explicit public data processing and weight redistribution consent required")
    if (type(reward_credits) is not int or not 0 <= reward_credits <= 1_000_000
            or not isinstance(owner_user_id, str) or not owner_user_id.strip()):
        raise ValueError("Invalid evaluation plan owner or reward")
    identity, payloads = _payloads(checkpoint, shards)
    plan = {"format": "ai-home-evaluation-plan-v1", "checkpoint_sha256": identity,
            "server": normalize_base_url(server), "owner_user_id": owner_user_id,
            "reward_credits": reward_credits, "payloads": payloads,
            "allow_public_node_processing": True, "weights_public_and_redistributable": True}
    plan["sha256"] = canonical_hash(plan)
    raw = json.dumps(plan, sort_keys=True).encode("utf-8")
    if len(raw) > 16_000_000:
        raise ValueError("Evaluation plan exceeds byte budget")
    publish_bytes(output, raw)
    return plan


def advance(client, plan_path, checkpoint, output):
    with Path(plan_path).open("rb") as source:
        raw = source.read(16_000_001)
    if len(raw) > 16_000_000:
        raise ValueError("Evaluation plan exceeds byte budget")
    plan = json.loads(raw)
    digest = plan.pop("sha256")
    if plan.get("format") != "ai-home-evaluation-plan-v1" or canonical_hash(plan) != digest:
        raise ValueError("Evaluation plan integrity failure")
    if (plan.get("allow_public_node_processing") is not True
            or plan.get("weights_public_and_redistributable") is not True):
        raise ValueError("Missing evaluation consent")
    if normalize_base_url(str(client.base_url)) != plan["server"]:
        raise ValueError("Evaluation server mismatch")
    identity, payloads = _payloads(checkpoint, [p["shard"]["records"] for p in plan["payloads"]])
    if identity != plan["checkpoint_sha256"] or payloads != plan["payloads"]:
        raise ValueError("Evaluation checkpoint or payload changed")
    response = client.get("/api/v1/users/me")
    response.raise_for_status()
    if response.json()["user_id"] != plan["owner_user_id"]:
        raise ValueError("Evaluation owner mismatch")
    jobs = []
    for index, payload in enumerate(payloads):
        response = client.post("/api/v1/jobs/foundation/evaluate", json={
            "payload": payload, "allow_public_node_processing": True,
            "reward_credits": plan["reward_credits"], "idempotency_key": f"evaluation-{digest}-{index}"})
        response.raise_for_status()
        jobs.append(response.json())
    if any(job["status"] in {"failed", "cancelled"} for job in jobs):
        raise ValueError("Evaluation job failed or cancelled; retry will not create replacements")
    if not all(job["status"] == "completed" for job in jobs):
        return {"state": "waiting", "jobs": [{"id": job["job_id"], "status": job["status"]} for job in jobs]}
    artifacts = []
    for payload, job in zip(payloads, jobs):
        if not verified_evaluation_result(payload, job["result"]):
            raise ValueError("Evaluation result failed independent verification")
        artifacts.append(json.loads(job["result"]["artifact_json"]))
    tokens = sum(a["tokens"] for a in artifacts)
    report = {"format": "ai-home-distributed-loss-report-v1", "state": "evaluated",
              "checkpoint_sha256": identity, "plan_sha256": digest, "tokens": tokens,
              "mean_loss": sum(a["tokens"] * a["mean_loss"] for a in artifacts) / tokens,
              "shards": artifacts, "job_ids": [job["job_id"] for job in jobs], "promoted": False,
              "scope": "Public causal loss; not language capability or independent hidden evaluation"}
    raw = json.dumps(report, sort_keys=True, allow_nan=False).encode("utf-8")
    try:
        publish_bytes(output, raw)
    except FileExistsError:
        with Path(output).open("rb") as source:
            if source.read(len(raw) + 1) != raw:
                raise ValueError("Existing evaluation report differs") from None
    return report
