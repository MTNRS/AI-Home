"""Restartable public loss-evaluation plans with checkpoint lineage checks."""
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ai_home.immutable_artifacts import publish_bytes
from ai_home.urls import normalize_base_url
from .checkpoint import load_checkpoint
from .data import canonical_hash, load_records, prepare_records
from .distributed import GradientPayload, encode_tensors
from .distributed_evaluation import verified_evaluation_result
from .tasks import PublicRecord


def dataset_shards(path, shard_count):
    """Partition a public holdout deterministically, without sampling or truncation."""
    if type(shard_count) is not int or not 1 <= shard_count <= 8:
        raise ValueError("Require one to eight evaluation shards")
    records, _ = load_records(path, max_bytes=1_000_000)
    if not 2 * shard_count <= len(records) <= 32 * shard_count:
        raise ValueError("Each evaluation shard requires two to 32 distinct documents")
    return [records[index::shard_count] for index in range(shard_count)]


def _payloads(checkpoint, shards):
    if not isinstance(shards, list) or not 1 <= len(shards) <= 8:
        raise ValueError("Require one to eight public evaluation shards")
    prepared, seen = [], set()
    for index, shard in enumerate(shards, start=1):
        if not isinstance(shard, list) or not 2 <= len(shard) <= 32:
            raise ValueError(f"Evaluation shard {index} requires two to 32 documents")
        records = prepare_records(shard)
        documents = {r["sha256"] for r in records}
        if documents & seen:
            raise ValueError(f"Evaluation shard {index} has duplicate documents across shards")
        if sum(len(r["text"].encode("utf-8")) for r in records) > 4096:
            raise ValueError(f"Evaluation shard {index} exceeds 4096 UTF-8 input bytes; repartition the holdout")
        public = [PublicRecord.model_validate({key: value for key, value in r.items()
                  if key in PublicRecord.model_fields}).model_dump() for r in records]
        prepared.append((records, documents, public))
        seen.update(documents)
    model, tokenizer, state = load_checkpoint(checkpoint, max_bytes=10_000_000)
    if state.get("data_visibility") != "public" or "training_document_hashes" not in state:
        raise ValueError("Evaluation requires a public checkpoint with training lineage")
    trained = set(state["training_document_hashes"])
    trained_groups = set(state.get("training_split_groups", []))
    parameters = encode_tensors(dict(model.named_parameters()))
    payloads = []
    for records, documents, public in prepared:
        groups = {r["split_group_sha256"] for r in records if "split_group_sha256" in r}
        if groups and "training_split_groups" not in state:
            raise ValueError("Grouped evaluation requires checkpoint group lineage")
        if documents & trained or groups & trained_groups:
            raise ValueError("Evaluation contamination or duplicate documents across shards")
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


def supervise(client, plan_path, checkpoint, output, *, wait_seconds, poll_seconds=5, cancel=None):
    from .job_observer import observe
    return observe(lambda: advance(client, plan_path, checkpoint, output), complete_state="evaluated",
                   wait_seconds=wait_seconds, poll_seconds=poll_seconds, cancel=cancel)


def main():
    import argparse
    import os
    import httpx
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Create an immutable public evaluation plan")
    inputs = plan.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--shards", type=Path, help="JSON array of record lists")
    inputs.add_argument("--dataset", type=Path, help="Authorized public holdout knowledge dataset")
    plan.add_argument("--shard-count", type=int, help="Number of shards for --dataset (1 to 8)")
    plan.add_argument("--owner", required=True)
    plan.add_argument("--reward-credits", type=int, default=0)
    plan.add_argument("--allow-public-node-processing", action="store_true")
    plan.add_argument("--weights-public-and-redistributable", action="store_true")
    resume = commands.add_parser("advance", help="Submit or collect once; safe to repeat after restart")
    resume.add_argument("--plan", type=Path, required=True)
    resume.add_argument("--wait-seconds", type=int, default=0)
    resume.add_argument("--poll-seconds", type=int, default=5)
    for command in (plan, resume):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--server", required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        if args.dataset is not None:
            if args.shard_count is None:
                parser.error("--dataset requires --shard-count")
            shards = dataset_shards(args.dataset, args.shard_count)
        else:
            if args.shard_count is not None:
                parser.error("--shard-count requires --dataset")
            with args.shards.open("rb") as source:
                raw = source.read(1_000_001)
            if len(raw) > 1_000_000:
                raise ValueError("Evaluation shards file exceeds input budget")
            shards = json.loads(raw)
        result = create_plan(args.checkpoint, shards, args.output,
            server=args.server, owner_user_id=args.owner, reward_credits=args.reward_credits,
            allow_public_node_processing=args.allow_public_node_processing,
            weights_public_and_redistributable=args.weights_public_and_redistributable)
        print(json.dumps({"state": "planned", "plan_sha256": result["sha256"],
                          "shards": len(result["payloads"])}))
    else:
        token = os.environ.get("AI_HOME_COORDINATOR_TOKEN", "")
        if not token:
            raise ValueError("Set AI_HOME_COORDINATOR_TOKEN for the authenticated owner")
        with httpx.Client(base_url=normalize_base_url(args.server), timeout=60, trust_env=False,
                          headers={"Authorization": f"Bearer {token}"}, follow_redirects=False) as client:
            try:
                if args.wait_seconds:
                    result = supervise(client, args.plan, args.checkpoint, args.output,
                                       wait_seconds=args.wait_seconds, poll_seconds=args.poll_seconds)
                else:
                    result = advance(client, args.plan, args.checkpoint, args.output)
            except KeyboardInterrupt:
                result = {"state": "interrupted", "jobs_cancelled": False}
            print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
