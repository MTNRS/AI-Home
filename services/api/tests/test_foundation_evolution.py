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


def test_evaluation_rejects_known_training_families_before_generation(tmp_path, monkeypatch):
    import ai_home.foundation.evaluation as module
    torch.set_num_threads(1)
    training = [{**r, 'split_group_sha256': 'a' * 64} for r in records('train')]
    validation = [{**r, 'split_group_sha256': 'b' * 64} for r in records('validation')]
    parent = tmp_path / 'grouped.pt'
    train(training, parent, FoundationConfig(vocab_size=262, width=16, layers=1, heads=2,
          kv_heads=1, hidden=32, context=16), TrainingConfig(steps=1, warmup=0), 'cpu',
          validation_records=validation)
    suite = {'version': 'group-test', 'source': 'authored fixture', 'license': 'Apache-2.0',
             'cases': [{'id': 'one', 'category': 'mathematics', 'prompt': '1+1=', 'answers': ['2']}]}
    assert ProbeSuite.model_validate(suite).model_dump() == suite
    suite['cases'][0]['split_group_sha256'] = 'a' * 64
    calls = []
    original = module.stream_tokens
    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'stream_tokens', observed)
    with pytest.raises(ValueError, match='Probe family overlaps'):
        evaluate_checkpoint(parent, suite, device='cpu', max_new_tokens=2)
    assert not calls
    suite['cases'][0]['split_group_sha256'] = 'c' * 64
    result = evaluate_checkpoint(parent, suite, device='cpu', max_new_tokens=2)
    assert calls and result['grouped_probe_count'] == 1 and result['group_lineage_available']
    suite['cases'][0]['split_group_sha256'] = 'invalid'
    with pytest.raises(ValueError):
        ProbeSuite.model_validate(suite)


@pytest.mark.parametrize("disk_sequences", [False, True])
def test_evolution_trains_two_real_candidates_and_reports_evidence(tmp_path, disk_sequences):
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
    from dataclasses import asdict
    from ai_home.foundation.evolution_job import run_evolution
    suite_path, training_path = tmp_path / "suite.json", tmp_path / "training.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    training_path.write_text(json.dumps(asdict(settings)), encoding="utf-8")
    report = run_evolution(parent, suite_path, paths["holdout"], paths["old"], paths["new"],
                           training_path, tmp_path / "experiment", device="cpu", max_new_tokens=4,
                           disk_sequences=disk_sequences)
    assert report["proposal"]["disk_sequences"] is disk_sequences
    for arm in ("control", "experiment"):
        assert report[arm]["disk_sequences"] is disk_sequences
        cache = tmp_path / "experiment" / arm / "sequence-cache"
        assert cache.exists() is disk_sequences
        if disk_sequences:
            for split in ("training/train", "training/validation", "training/replay", "baseline", "candidate"):
                assert (cache / split / "manifest.json").is_file()
    import threading
    from ai_home.foundation.training import TrainingCancelled
    from ai_home.foundation.checkpoint import load_checkpoint
    cancellation = threading.Event()
    stages = []
    def progress(item):
        stages.append(item["stage"])
        if item["stage"] == "control_training" and item.get("step") == 1:
            cancellation.set()
    interrupted = tmp_path / "interrupted"
    with pytest.raises(TrainingCancelled):
        run_experiment(parent, interrupted, suite, paths["holdout"], paths["old"],
                       dataset=paths["new"], training=settings, device="cpu", max_new_tokens=2,
                       cancel=cancellation, on_progress=progress)
    assert "baseline_evaluation" in stages and "control_training" in stages
    checkpoint = load_checkpoint(interrupted / "control" / "candidate.pt")[2]
    assert checkpoint["step"] == 1 and checkpoint["stop_reason"] == "cancelled"
    assert not (interrupted / "result.json").exists()
    assert not (interrupted / "control" / "report.json").exists()
    assert not (interrupted / "experiment").exists()
    state = json.loads((interrupted / "interrupted.json").read_text())
    assert state["stage"] == "control_training" and state["promoted"] is False
    with pytest.raises(TrainingCancelled):
        run_experiment(parent, tmp_path / "never", suite, paths["holdout"], paths["old"],
                       dataset=paths["new"], training=settings, cancel=cancellation)
    assert not (tmp_path / "never").exists()
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


def test_evaluation_job_progress_cancellation_and_exclusive_report(tmp_path):
    import threading
    from ai_home.foundation.evaluation import EvaluationCancelled
    from ai_home.foundation.evaluation_job import run_evaluation
    from ai_home.foundation.reports import read_report
    checkpoint = tmp_path / "model.pt"
    config = FoundationConfig(vocab_size=280, width=32, layers=1, heads=4, kv_heads=2, hidden=64, context=128)
    train(records("job"), checkpoint, config, TrainingConfig(steps=1, warmup=0), "cpu")
    suite = Path(__file__).resolve().parents[3] / "docs/evaluation-probes-v1.json"
    output = tmp_path / "evaluation.json"
    progress = []
    result = run_evaluation(checkpoint, suite, output, max_new_tokens=2, on_probe=progress.append)
    assert len(progress) == len(result["results"]) == 10
    assert progress[-1]["completed"] == 10
    assert read_report(output)["report"] == result
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        run_evaluation(checkpoint, suite, output)
    assert output.read_bytes() == original
    cancellation = threading.Event()
    cancelled = tmp_path / "cancelled.json"
    with pytest.raises(EvaluationCancelled):
        run_evaluation(checkpoint, suite, cancelled, max_new_tokens=2, cancel=cancellation,
                       on_probe=lambda item: cancellation.set())
    assert not cancelled.exists()
    with pytest.raises(EvaluationCancelled):
        run_evaluation(checkpoint, suite, cancelled, cancel=cancellation)


def test_evaluation_identity_tracks_loaded_weights_not_replaced_path(tmp_path):
    import hashlib
    from ai_home.foundation.checkpoint import load_checkpoint
    checkpoint = tmp_path / "original.pt"
    config = FoundationConfig(vocab_size=280, width=16, layers=1, heads=2, kv_heads=1, hidden=32, context=64)
    train(records("identity"), checkpoint, config, TrainingConfig(steps=1, warmup=0), "cpu")
    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert load_checkpoint(checkpoint)[0].checkpoint_sha256 == original_hash
    suite = {"version":"identity-v1", "source":"authored test", "license":"Apache-2.0",
             "cases":[{"id":"one", "category":"mathematics", "prompt":"1+1?", "answers":["2"]},
                      {"id":"two", "category":"mathematics", "prompt":"2+2?", "answers":["4"]}]}
    def replace_path(_):
        checkpoint.write_bytes(b"Different bytes after model loading")
    report = evaluate_checkpoint(checkpoint, suite, device="cpu", max_new_tokens=2, on_probe=replace_path)
    assert report["checkpoint_sha256"] == original_hash
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() != original_hash
    assert len(report["results"]) == 2


@pytest.mark.parametrize("changed", ["new", "old", "holdout"])
def test_evolution_refuses_data_changes_between_arms(tmp_path, changed):
    torch.set_num_threads(1)
    paths = {}
    for name in ("old", "new", "holdout"):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps({"format_version": 1, "records": records(name)}), encoding="utf-8")
    config = FoundationConfig(vocab_size=280, width=16, layers=1, heads=2,
                              kv_heads=1, hidden=32, context=128)
    settings = TrainingConfig(steps=1, warmup=0)
    parent = tmp_path / "parent.pt"
    train(records("old"), parent, config, settings, "cpu", validation_records=records("holdout"))
    suite = {"version": "data-change-v1", "source": "authored test", "license": "Apache-2.0", "cases": [
        {"id": "sum", "category": "mathematics", "prompt": "2+2=", "answers": ["4"]}]}
    stages = []
    def replace_data(item):
        stages.append(item["stage"])
        if item["stage"] == "experiment_training" and "step" not in item:
            paths[changed].write_text(json.dumps({"format_version": 1,
                "records": records("replacement " + changed)}), encoding="utf-8")
    output = tmp_path / "experiment"
    with pytest.raises(ValueError, match="invariants differ"):
        run_experiment(parent, output, suite, paths["holdout"], paths["old"],
                       dataset=paths["new"], training=settings, device="cpu",
                       max_new_tokens=2, on_progress=replace_data)
    assert "control_evaluation" in stages
    assert "experiment_evaluation" not in stages
    assert not (output / "result.json").exists()
    invalid = json.loads((output / "invalid-experiment.json").read_text(encoding="utf-8"))
    assert invalid["promoted"] is False
    assert "dataset_sha256" in invalid["different_fields"]
    if changed in ("old", "holdout"):
        assert ("replay_sha256" if changed == "old" else "validation_sha256") in invalid["different_fields"]


@pytest.mark.parametrize("arm", ["control", "experiment"])
def test_evolution_rejects_checkpoint_replacement_before_generation(tmp_path, arm):
    torch.set_num_threads(1)
    paths = {}
    for name in ("old", "new", "holdout"):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps({"format_version": 1, "records": records(name)}), encoding="utf-8")
    config = FoundationConfig(vocab_size=280, width=16, layers=1, heads=2,
                              kv_heads=1, hidden=32, context=128)
    settings = TrainingConfig(steps=1, warmup=0)
    parent = tmp_path / "parent.pt"
    train(records("old"), parent, config, settings, "cpu", validation_records=records("holdout"))
    suite = {"version": "identity-v1", "source": "authored test", "license": "Apache-2.0", "cases": [
        {"id": "sum", "category": "mathematics", "prompt": "2+2=", "answers": ["4"]}]}
    output = tmp_path / "experiment"
    replaced = []
    def replace_checkpoint(item):
        if item["stage"] == arm + "_evaluation" and not replaced:
            (output / arm / "candidate.pt").write_bytes(parent.read_bytes())
            replaced.append(True)
    with pytest.raises(ValueError, match="checkpoint identities differ"):
        run_experiment(parent, output, suite, paths["holdout"], paths["old"],
                       dataset=paths["new"], training=settings, device="cpu",
                       max_new_tokens=2, on_progress=replace_checkpoint)
    assert replaced == [True]
    assert not (output / "result.json").exists()
    invalid = json.loads((output / "invalid-experiment.json").read_text(encoding="utf-8"))
    assert invalid["stage"] == arm and invalid["promoted"] is False
    assert invalid["loss_checkpoint_sha256"] != invalid["generation_checkpoint_sha256"]
    evaluation = json.loads((output / f"{arm}-evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["checkpoint_sha256"] == invalid["generation_checkpoint_sha256"]
