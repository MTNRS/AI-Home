import json
import copy
from pathlib import Path

import pytest

from ai_home.foundation.reports import read_report


DOCS = Path(__file__).parents[3] / "docs"


def test_adapter_report_metrics_lifecycle_and_resources(tmp_path):
    report = read_report(DOCS / "ADAPTER_RESOURCES_CUDA.json")["report"]
    assert report["step"] == 20 and report["trainable_parameters"] == 448
    target = tmp_path / "adapter.json"
    for field, value in (("step", True), ("checkpoint", "../other"),
                         ("promoted", True), ("cancelled", True),
                         ("training_complete", False), ("candidate", None), ("objective", "sft")):
        changed = {**report, field: value}
        target.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            read_report(target)
    for group, field, value in (("candidate", "tokens", 1), ("baseline", "loss", -1),
            ("run_resources", "new_steps", 21), ("run_resources", "tokens_per_second", True),
            ("run_resources", "cuda_peak_allocated_bytes", -1)):
        changed = copy.deepcopy(report)
        changed[group][field] = value
        target.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            read_report(target)
    cancelled = {**report, "cancelled": True, "candidate": None}
    target.write_text(json.dumps(cancelled), encoding="utf-8")
    assert read_report(target)["report"]["candidate"] is None


def test_preference_reports_and_cancelled_final_evaluation(tmp_path):
    report = json.loads((DOCS / "PREFERENCE_SMOKE_CPU.json").read_text(encoding="utf-8"))["training"]
    path = tmp_path / "report.json"
    for cancelled in (False, True):
        report["cancelled"] = cancelled
        if cancelled:
            report["candidate"] = None
        path.write_text(json.dumps(report), encoding="utf-8")
        assert read_report(path)["report"] == report


@pytest.mark.parametrize("mutation", ["pairs", "fraction", "promotion", "missing", "step", "visibility"])
def test_reject_inconsistent_preference_reports(tmp_path, mutation):
    report = json.loads((DOCS / "PREFERENCE_SMOKE_CPU.json").read_text(encoding="utf-8"))["training"]
    if mutation == "pairs":
        report["candidate"]["pairs"] += 1
    elif mutation == "fraction":
        report["candidate"]["preferred_fraction"] = 1.1
    elif mutation == "promotion":
        report["promoted"] = True
    elif mutation == "missing":
        report["candidate"] = None
    elif mutation == "step":
        report["step"] += 1
    else:
        report["data_visibility"] = "public"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        read_report(path)


@pytest.mark.parametrize("filename", ["EVALUATION_BASELINE_CPU.json", "EVALUATION_CONTROL_CPU.json",
                                      "EVALUATION_EXPERIMENT_CPU.json", "EVOLUTION_SMOKE_CPU.json"])
def test_inspect_recorded_real_experiments(filename):
    result = read_report(DOCS / filename)
    assert len(result["file_sha256"]) == 64
    assert result["file_bytes"] == (DOCS / filename).stat().st_size
    if "results" in result["report"]:
        assert len(result["report"]["results"]) == 10


def test_report_rejects_inconsistent_metrics_and_unbounded_input(tmp_path):
    report = json.loads((DOCS / "EVALUATION_BASELINE_CPU.json").read_text())
    report["results"][0]["passed"] = True
    target = tmp_path / "bad.json"
    target.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        read_report(target)
    with pytest.raises(ValueError, match="budget"):
        read_report(target, max_bytes=10)
    for invalid in ['{"format":"ai-home-evaluation-v1","value":NaN}',
                    '{"format":"ai-home-evaluation-v1","value":1e999}',
                    '{"format":"unrecognized"}']:
        target.write_text(invalid)
        with pytest.raises(ValueError):
            read_report(target)


def test_inspect_real_distributed_loss_and_reject_inconsistent_shards(tmp_path):
    import copy
    document = read_report(DOCS / "DISTRIBUTED_EVALUATION_REPORT.json")
    report = document["report"]
    assert report["tokens"] == 468 and len(report["shards"]) == 2
    assert "not independently verified" in document["notice"]
    for change in ("tokens", "loss", "base", "documents", "jobs", "promotion", "boolean", "huge_tokens", "huge_loss"):
        altered = copy.deepcopy(report)
        if change == "tokens":
            altered["tokens"] += 1
        elif change == "loss":
            altered["mean_loss"] += 1
        elif change == "base":
            altered["shards"][0]["base_sha256"] = "0" * 64
        elif change == "documents":
            altered["shards"][1]["document_hashes"] = altered["shards"][0]["document_hashes"]
        elif change == "jobs":
            altered["job_ids"][1] = altered["job_ids"][0]
        elif change == "promotion":
            altered["promoted"] = True
        elif change == "huge_tokens":
            altered["shards"][0]["tokens"] = 10**400
        elif change == "huge_loss":
            altered["mean_loss"] = 10**400
        else:
            altered["shards"][0]["tokens"] = True
        path = tmp_path / "changed.json"
        path.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(ValueError):
            read_report(path)
