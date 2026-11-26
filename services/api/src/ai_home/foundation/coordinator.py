"""Restartable single-round coordinator using durable server job idempotency."""
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time

import httpx

from ai_home.immutable_artifacts import publish_bytes
from ai_home.urls import normalize_base_url
from .checkpoint import load_checkpoint
from .data import canonical_hash, prepare_records
from .distributed import GradientPayload, encode_tensors, parameter_fingerprint
from .distributed_round import commit_round


def create_plan(parent, payloads, output, *, server, owner_user_id, reward_credits=0,
                allow_public_node_processing=False, scheduling=None):
    if allow_public_node_processing is not True:
        raise ValueError("Explicit public node processing consent required")
    if (not isinstance(payloads, list) or not 2 <= len(payloads) <= 8
            or type(reward_credits) is not int or not 0 <= reward_credits <= 1_000_000
            or not isinstance(owner_user_id, str) or not owner_user_id.strip()):
        raise ValueError("Invalid coordinator plan")
    model, tokenizer, state = load_checkpoint(parent, max_bytes=10_000_000)
    if (state.get("data_visibility") != "public" or state.get("training_mode") != "verified-gradient-rounds-v1"
            or "optimizer" not in state or type(state.get("step")) is not int):
        raise ValueError("Coordinator requires a public distributed parent")
    current = parameter_fingerprint(asdict(model.config), tokenizer.to_dict(), encode_tensors(dict(model.named_parameters())))
    validated, seen = [], set()
    for payload in payloads:
        value = GradientPayload.model_validate(payload).model_dump()
        if parameter_fingerprint(value["config"], value["shard"]["tokenizer"], value["parameters"]) != current:
            raise ValueError("Shard does not match parent model/tokenizer")
        documents = {record["sha256"] for record in prepare_records(value["shard"]["records"])}
        if documents & (seen | set(state.get("validation_document_hashes", []))):
            raise ValueError("Overlapping shards or fixed validation data")
        seen.update(documents)
        validated.append(value)
    plan = {"format": "ai-home-coordinator-plan-v1", "server": normalize_base_url(server),
            "owner_user_id": owner_user_id, "parent_sha256": model.checkpoint_sha256,
            "payloads": validated, "reward_credits": reward_credits,
            "allow_public_node_processing": True}
    if scheduling is not None:
        from ai_home.schemas import GradientSchedulingRequirements
        plan["scheduling"] = GradientSchedulingRequirements.model_validate(scheduling).model_dump(exclude_none=True)
    plan["sha256"] = canonical_hash(plan)
    raw = json.dumps(plan, sort_keys=True).encode("utf-8")
    if len(raw) > 16_000_000:
        raise ValueError("Coordinator plan exceeds byte budget")
    publish_bytes(output, raw)
    return plan


def advance(client, plan_path, parent, output):
    """One resumable pass: submit/find jobs, or commit when all are complete.

    No token is stored. Re-running this operation after lost HTTP replies uses the
    exact same owner-scoped job keys, preserving server reservations/rewards.
    """
    with Path(plan_path).open("rb") as source:
        raw = source.read(16_000_001)
    if len(raw) > 16_000_000:
        raise ValueError("Coordinator plan exceeds byte budget")
    plan = json.loads(raw)
    digest = plan.pop("sha256")
    if plan.get("format") != "ai-home-coordinator-plan-v1" or canonical_hash(plan) != digest:
        raise ValueError("Coordinator plan integrity failure")
    if normalize_base_url(str(client.base_url)) != plan["server"]:
        raise ValueError("Coordinator server mismatch")
    response = client.get("/api/v1/users/me")
    response.raise_for_status()
    if response.json()["user_id"] != plan["owner_user_id"]:
        raise ValueError("Coordinator owner mismatch")
    model, _, _ = load_checkpoint(parent, max_bytes=10_000_000)
    if model.checkpoint_sha256 != plan["parent_sha256"]:
        raise ValueError("Coordinator parent changed")
    jobs = []
    for index, payload in enumerate(plan["payloads"]):
        response = client.post("/api/v1/jobs/foundation/gradient", json={
            "payload": payload, "allow_public_node_processing": plan["allow_public_node_processing"],
            "reward_credits": plan["reward_credits"], "idempotency_key": f"coordinator-{digest}-{index}",
            **({"scheduling": plan["scheduling"]} if "scheduling" in plan else {})})
        response.raise_for_status()
        jobs.append(response.json())
    if any(job["status"] in {"failed", "cancelled"} for job in jobs):
        raise ValueError("Coordinator job failed or was cancelled; no new jobs created on retry")
    if not all(job["status"] == "completed" for job in jobs):
        return {"state": "waiting", "jobs": [{"id": job["job_id"], "status": job["status"]} for job in jobs]}
    result = commit_round(parent, [{"payload": payload, "result": job["result"]}
                                  for payload, job in zip(plan["payloads"], jobs)], output)
    return {"state": "committed", "step": result["step"], "round_request_sha256": result["round_request_sha256"]}


def supervise(client, plan_path, parent, output, *, wait_seconds, poll_seconds=5, cancel=None):
    """Bounded restartable polling; stopping the observer does not cancel funded jobs."""
    if (type(wait_seconds) is not int or not 1 <= wait_seconds <= 3600
            or type(poll_seconds) is not int or not 1 <= poll_seconds <= 60):
        raise ValueError("Invalid coordinator wait bounds")
    cancellation = cancel if cancel is not None else threading.Event()
    deadline = time.monotonic() + wait_seconds
    last, attempts, transport_errors = None, 0, 0
    while not cancellation.is_set():
        if time.monotonic() >= deadline:
            break
        attempts += 1
        try:
            last = advance(client, plan_path, parent, output)
        except httpx.TransportError:
            # A response may be lost after creation; replay the same immutable plan.
            # HTTP authentication/policy/status failures are deliberately not retried.
            transport_errors += 1
        else:
            if last["state"] == "committed":
                return {**last, "attempts": attempts, "transport_errors": transport_errors}
        cancellation.wait(max(0, min(poll_seconds, deadline - time.monotonic())))
    return {"state": "stopped" if cancellation.is_set() else "timed_out", "last_observation": last,
            "attempts": attempts, "transport_errors": transport_errors,
            "jobs_cancelled": False}


def main():
    import argparse
    import os
    import httpx
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("plan")
    create.add_argument("parent", type=Path)
    create.add_argument("payloads", type=Path)
    create.add_argument("output", type=Path)
    create.add_argument("--server", required=True)
    create.add_argument("--owner", required=True)
    create.add_argument("--reward-credits", type=int, default=0)
    create.add_argument("--allow-public-node-processing", action="store_true")
    resume = commands.add_parser("advance")
    resume.add_argument("plan", type=Path)
    resume.add_argument("parent", type=Path)
    resume.add_argument("output", type=Path)
    resume.add_argument("--server", required=True)
    resume.add_argument("--wait-seconds", type=int, default=0)
    resume.add_argument("--poll-seconds", type=int, default=5)
    args = parser.parse_args()
    if args.command == "plan":
        with args.payloads.open("rb") as source:
            raw = source.read(16_000_001)
        if len(raw) > 16_000_000:
            raise ValueError("Payload file exceeds byte budget")
        result = create_plan(args.parent, json.loads(raw), args.output, server=args.server,
                             owner_user_id=args.owner, reward_credits=args.reward_credits,
                             allow_public_node_processing=args.allow_public_node_processing)
        print(json.dumps({"plan_sha256": result["sha256"], "state": "planned"}))
    else:
        token = os.environ.get("AI_HOME_COORDINATOR_TOKEN", "")
        if not token:
            raise ValueError("Set AI_HOME_COORDINATOR_TOKEN for the authenticated owner")
        with httpx.Client(base_url=normalize_base_url(args.server), timeout=60, trust_env=False,
                          headers={"Authorization": f"Bearer {token}"}, follow_redirects=False) as client:
            try:
                if args.wait_seconds:
                    result = supervise(client, args.plan, args.parent, args.output,
                                       wait_seconds=args.wait_seconds, poll_seconds=args.poll_seconds)
                else:
                    result = advance(client, args.plan, args.parent, args.output)
            except KeyboardInterrupt:
                result = {"state": "interrupted", "jobs_cancelled": False}
            print(json.dumps(result))


if __name__ == "__main__":
    main()
