"""Bounded local experiment report inspection; never executes report content."""
import hashlib
import json
import math
import re
from pathlib import Path


FORMATS = {"ai-home-evaluation-v1", "ai-home-evolution-result-v1", "ai-home-distributed-loss-report-v1"}


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
    else:
        if (report.get("decision") not in {"reject", "propose_acceptance"}
                or report.get("promoted") is not False
                or not isinstance(report.get("proposal"), dict)
                or not isinstance(report.get("reason"), str)):
            raise ValueError("Invalid evolution result")
    return {"file_sha256": hashlib.sha256(raw).hexdigest(), "file_bytes": len(raw),
            "notice": "Local report content; checkpoint identity and execution are not independently verified",
            "report": report}


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
