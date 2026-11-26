"""Bounded local experiment report inspection; never executes report content."""
import hashlib
import json
from pathlib import Path


FORMATS = {"ai-home-evaluation-v1", "ai-home-evolution-result-v1"}


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
    else:
        if (report.get("decision") not in {"reject", "propose_acceptance"}
                or report.get("promoted") is not False
                or not isinstance(report.get("proposal"), dict)
                or not isinstance(report.get("reason"), str)):
            raise ValueError("Invalid evolution result")
    return {"file_sha256": hashlib.sha256(raw).hexdigest(), "file_bytes": len(raw),
            "notice": "Local report content; checkpoint identity and execution are not independently verified",
            "report": report}
