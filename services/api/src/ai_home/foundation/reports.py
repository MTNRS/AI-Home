"""Bounded local experiment report inspection; never executes report content."""
import hashlib
import json
import math
import re
from pathlib import Path


FORMATS = {"ai-home-evaluation-v1", "ai-home-evolution-result-v1", "ai-home-distributed-loss-report-v1"}
FORMATS.add("ai-home-preference-training-v1")
FORMATS.add("ai-home-adapter-training-v1")


def read_report(path, max_bytes=2_000_000):
    with Path(path).open("rb") as source:
        raw = source.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("Report exceeds read budget")
    def invalid_constant(value):
        raise ValueError("Non-finite report value: " + value)
    report = json.loads(raw.decode("utf-8"), parse_constant=invalid_constant)
    if not isinstance(report, dict) or report.get("format") not in FORMATS:
        raise ValueError("Unsupported evaluation/evolution report")
    # Reject overflow floats too (e.g. 1e999), not just JSON NaN literals.
    json.dumps(report, allow_nan=False)
    if report["format"] == "ai-home-evaluation-v1":
        rows = report.get("results")
        if not isinstance(rows, list) or not rows or len(rows) > 1000:
            raise ValueError("Invalid evaluation results")
        ids, categories = set(), {}
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                    or row["id"] in ids or not isinstance(row.get("category"), str)
                    or type(row.get("passed")) is not bool
                    or type(row.get("prompt_truncated")) is not bool):
                raise ValueError("Invalid evaluation row")
            if row["passed"] and row["prompt_truncated"]:
                raise ValueError("Truncated probe cannot pass")
            ids.add(row["id"])
            group = categories.setdefault(row["category"], {"passed": 0, "total": 0})
            group["total"] += 1
            group["passed"] += row["passed"]
        for group in categories.values():
            group["accuracy"] = group["passed"] / group["total"]
        if report.get("categories") != categories:
            raise ValueError("Evaluation category totals disagree with results")
        if report.get("all_prompts_fit") != (not any(r["prompt_truncated"] for r in rows)):
            raise ValueError("Evaluation truncation summary disagrees with results")
    elif report["format"] == "ai-home-distributed-loss-report-v1":
        validate_distributed_loss(report)
    elif report["format"] == "ai-home-preference-training-v1":
        validate_preference_report(report)
    elif report["format"] == "ai-home-adapter-training-v1":
        validate_adapter_report(report)
    else:
        if (report.get("decision") not in {"reject", "propose_acceptance"}
                or report.get("promoted") is not False
                or not isinstance(report.get("proposal"), dict)
                or not isinstance(report.get("reason"), str)):
            raise ValueError("Invalid evolution result")
    return {"file_sha256": hashlib.sha256(raw).hexdigest(), "file_bytes": len(raw),
            "notice": "Local report content; checkpoint identity and execution are not independently verified",
            "report": report}


def validate_adapter_report(report):
    """Validate displayed adapter metrics; never load weights or execute training."""
    def count(value, minimum=1):
        return type(value) is int and minimum <= value <= 2**63 - 1
    def number(value):
        return type(value) in (int, float) and 0 <= value <= 1e300
    identity = report.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("settings"), dict):
        raise ValueError("Invalid adapter identity")
    if not isinstance(identity.get("parent_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", identity["parent_sha256"]):
        raise ValueError("Invalid adapter parent digest")
    objective = report.get("objective", "causal")
    if objective not in ("causal", "sft") or identity["settings"].get("objective", "causal") != objective:
        raise ValueError("Inconsistent adapter objective")
    step, steps = report.get("step"), identity["settings"].get("steps")
    if (not count(step) or not count(steps) or step > steps or steps > 10000
            or report.get("checkpoint") != f"step-{step}"
            or type(report.get("cancelled")) is not bool
            or type(report.get("training_complete")) is not bool
            or report["training_complete"] != (step == steps)
            or not count(report.get("trainable_parameters"))
            or report.get("promoted") is not False
            or report.get("data_visibility") != "reviewed-training-only"):
        raise ValueError("Invalid adapter lifecycle")
    baseline, candidate = report.get("baseline"), report.get("candidate")
    for metrics in [baseline] + ([candidate] if candidate is not None else []):
        if (not isinstance(metrics, dict) or not count(metrics.get("tokens"))
                or not number(metrics.get("loss")) or not number(metrics.get("perplexity"))
                or metrics["perplexity"] < 1):
            raise ValueError("Invalid adapter metrics")
    if ((candidate is None) != report["cancelled"]
            or candidate is not None and candidate["tokens"] != baseline["tokens"]):
        raise ValueError("Inconsistent adapter holdout")
    resources = report.get("run_resources")
    if resources is not None:
        if (not isinstance(resources, dict) or not count(resources.get("new_steps"))
                or resources["new_steps"] > step or not count(resources.get("supervised_tokens"))
                or not count(resources.get("process_rss_bytes_after"))
                or not all(number(resources.get(key)) for key in
                           ("update_seconds", "tokens_per_second", "training_save_evaluation_seconds"))):
            raise ValueError("Invalid adapter resource metrics")
        for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
            if resources.get(key) is not None and not count(resources[key], 0):
                raise ValueError("Invalid adapter CUDA memory metric")


def validate_preference_report(report):
    """Check displayed metrics and lifecycle consistency, not execution authenticity."""
    def count(value):
        return type(value) is int and 1 <= value <= 10000
    def number(value):
        return type(value) in (int, float) and math.isfinite(value)
    identity = report.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("settings"), dict):
        raise ValueError("Invalid preference identity")
    for value in [report.get("candidate_checkpoint_sha256"),
                  *(identity.get(key) for key in ("parent_sha256", "pairs_sha256",
                    "training_sha256", "validation_sha256"))]:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Invalid preference digest")
    step, steps = report.get("step"), identity["settings"].get("steps")
    if (not count(step) or not count(steps) or step > steps
            or not count(report.get("training_pairs"))
            or type(report.get("cancelled")) is not bool
            or type(report.get("training_complete")) is not bool
            or report["training_complete"] != (step == steps)
            or report.get("promoted") is not False
            or report.get("data_visibility") != "reviewed-training-only"):
        raise ValueError("Invalid preference lifecycle")
    baseline, candidate = report.get("baseline"), report.get("candidate")
    for metrics in [baseline] + ([candidate] if candidate is not None else []):
        if (not isinstance(metrics, dict) or not count(metrics.get("pairs"))
                or not all(number(metrics.get(key)) for key in
                           ("mean_margin", "preferred_fraction", "mean_dpo_loss"))
                or not 0 <= metrics["preferred_fraction"] <= 1
                or metrics["mean_dpo_loss"] < 0):
            raise ValueError("Invalid preference metrics")
    if ((candidate is None) != report["cancelled"]
            or candidate is not None and candidate["pairs"] != baseline["pairs"]):
        raise ValueError("Inconsistent preference holdout")


def validate_distributed_loss(report):
    def digest(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    def count(value):
        return type(value) is int and 0 < value <= 2**63 - 1
    def loss(value):
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value) and value >= 0
        except OverflowError:
            return False
    shards, jobs = report.get("shards"), report.get("job_ids")
    if (report.get("state") != "evaluated" or report.get("promoted") is not False
            or not digest(report.get("checkpoint_sha256")) or not digest(report.get("plan_sha256"))
            or not isinstance(shards, list) or not 1 <= len(shards) <= 8
            or not isinstance(jobs, list) or len(jobs) != len(shards)
            or any(not isinstance(job, str) or not job for job in jobs)
            or len(set(jobs)) != len(jobs) or not count(report.get("tokens"))
            or not loss(report.get("mean_loss"))):
        raise ValueError("Invalid distributed loss report")
    seen, bases, tokens, weighted = set(), set(), 0, 0.0
    for shard in shards:
        if (not isinstance(shard, dict) or shard.get("format") != "ai-home-evaluation-shard-v1"
                or shard.get("objective") != "causal" or not count(shard.get("tokens"))
                or not loss(shard.get("mean_loss")) or not digest(shard.get("base_sha256"))
                or not digest(shard.get("dataset_sha256"))):
            raise ValueError("Invalid distributed loss shard")
        documents = shard.get("document_hashes")
        if (not isinstance(documents, list) or not 2 <= len(documents) <= 32
                or any(not digest(value) for value in documents)
                or len(set(documents)) != len(documents) or seen.intersection(documents)):
            raise ValueError("Duplicate or invalid evaluation documents")
        seen.update(documents)
        bases.add(shard["base_sha256"])
        tokens += shard["tokens"]
        weighted += shard["tokens"] * shard["mean_loss"]
    if (len(bases) != 1 or tokens != report["tokens"]
            or not math.isclose(weighted / tokens, report["mean_loss"], rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("Distributed loss summary disagrees with shards")
