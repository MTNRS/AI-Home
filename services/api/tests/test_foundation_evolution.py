import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ai_home.foundation.evaluation import evaluate_checkpoint, ProbeSuite
from ai_home.foundation.evolution import propose_decision, run_experiment
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.training import TrainingConfig, train


def records(prefix):
    return [{"source": "authored experiment fixture", "url": "local:tests", "license": "Apache-2.0",
             "date": "2026-11-26", "language": "es", "training_allowed": True,
             "privacy_reviewed": True, "visibility": "public", "text": f"{prefix} {i}: datos autorizados. " * 8}
            for i in range(4)]


def test_evolution_trains_two_real_candidates_and_reports_evidence(tmp_path):
    torch.set_num_threads(1)
    paths = {}
    for name in ("old", "new", "holdout"):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps({"format_version": 1, "records": records(name)}), encoding="utf-8")
    config = FoundationConfig(vocab_size=280, width=32, layers=2, heads=4, kv_heads=2, hidden=64, context=128)
    settings = TrainingConfig(steps=2, warmup=1)
    parent = tmp_path / "parent.pt"
    train(records("old"), parent, config, settings, "cpu", validation_records=records("holdout"))
    suite = json.loads((Path(__file__).resolve().parents[3] / "docs/evaluation-probes-v1.json").read_text(encoding="utf-8"))
    report = run_experiment(parent, tmp_path / "experiment", suite, paths["holdout"], paths["old"],
                            dataset=paths["new"], training=settings, device="cpu", max_new_tokens=4)
    assert report["promoted"] is False
    assert report["decision"] == "reject"
    assert report["control"]["embedding_delta_l2"] > 0
    assert report["experiment"]["embedding_delta_l2"] > 0
    assert report["control"]["candidate_sha256"] != report["experiment"]["candidate_sha256"]
    assert report["control"]["parent_sha256"] == report["experiment"]["parent_sha256"]
    first = evaluate_checkpoint(parent, suite, device="cpu", max_new_tokens=4)
    second = evaluate_checkpoint(parent, suite, device="cpu", max_new_tokens=4)
    assert first["all_prompts_fit"]
    assert first["categories"] == second["categories"]
    assert [r["generated"] for r in first["results"]] == [r["generated"] for r in second["results"]]
    assert all(r["generation_seconds"] > 0 for r in first["results"])
    # Verify the exact-match scorer against a known deterministic output, not a capability claim.
    scorer_fixture = copy.deepcopy(suite)
    scorer_fixture["cases"] = scorer_fixture["cases"][:1]
    scorer_fixture["cases"][0]["answers"] = [first["results"][0]["generated"]]
    assert evaluate_checkpoint(parent, scorer_fixture, max_new_tokens=4)["results"][0]["passed"]
    oversized = copy.deepcopy(suite)
    oversized["cases"][0]["prompt"] = "x" * 500
    assert not evaluate_checkpoint(parent, oversized, max_new_tokens=1)["all_prompts_fit"]
    contaminated = copy.deepcopy(suite)
    contaminated["cases"][0]["prompt"] = records("old")[0]["text"]
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_checkpoint(parent, contaminated, max_new_tokens=1)
    with pytest.raises(FileExistsError):
        run_experiment(parent, tmp_path / "experiment", suite, paths["holdout"], paths["old"],
                       dataset=paths["new"], training=settings)


def decision_report(passes):
    rows = [{"id": str(i), "category": "mathematics", "passed": value} for i, value in enumerate(passes)]
    return {"suite_sha256": "fixed", "generation": {"temperature": 0}, "device": "cpu",
            "torch_version": "test", "all_prompts_fit": True, "training_lineage_available": True,
            "results": rows, "categories": {"mathematics": {"passed": sum(passes),
            "total": len(passes), "accuracy": sum(passes) / len(passes)}}}


def test_decision_requires_measured_capability_gain_not_just_lower_loss():
    control = decision_report([True, False, False])
    candidate = decision_report([True, True, False])
    assert propose_decision(control, control, 5, 4, "mathematics")[0] == "reject"
    assert propose_decision(control, candidate, 5, 4, "mathematics")[0] == "propose_acceptance"
    # Aggregate accuracy increases, but one previously correct answer is lost.
    regression = decision_report([False, True, True])
    decision, reason = propose_decision(control, regression, 5, 4, "mathematics")
    assert decision == "reject" and "probes regressed: 0" in reason
    for loss in (float("nan"), float("inf")):
        assert propose_decision(control, candidate, loss, 4, "mathematics")[0] == "reject"
    for field in ("suite_sha256", "device", "torch_version", "generation"):
        invalid = copy.deepcopy(candidate)
        invalid[field] = "different"
        with pytest.raises(ValueError, match="protocols"):
            propose_decision(control, invalid, 5, 4, "mathematics")
    candidate["results"][0]["passed"] = False
    with pytest.raises(ValueError, match="disagree"):
        propose_decision(control, candidate, 5, 4, "mathematics")


def test_suite_rejects_ambiguous_or_executable_scoring():
    suite = {"version": "v1", "source": "authored", "license": "Apache-2.0",
             "cases": [{"id": "one", "category": "mathematics", "prompt": "1+1?", "answers": ["2"]}]}
    suite["cases"][0]["execute"] = "print('untrusted code')"
    with pytest.raises(ValueError):
        ProbeSuite.model_validate(suite)
