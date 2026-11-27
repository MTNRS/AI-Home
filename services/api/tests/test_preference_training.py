import copy
import json
import threading

import pytest
torch = pytest.importorskip("torch")
from ai_home.foundation.checkpoint import load_checkpoint
from ai_home.foundation.data import canonical_hash
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.preference_training import train_preferences
from ai_home.foundation.training import TrainingConfig, train
from test_preference_data import fixture
from test_foundation_continual import records


def inputs(tmp_path):
    torch.set_num_threads(1)
    parent = tmp_path / "parent.pt"
    train(records("own parent"), parent, FoundationConfig(vocab_size=262, width=16, layers=1,
        heads=2, kv_heads=1, hidden=32, context=64), TrainingConfig(steps=1, warmup=0), "cpu")
    export, review = fixture(tmp_path)
    base = export["records"]
    export["records"], review["pairs"] = [], []
    for index in range(12):
        rows = copy.deepcopy(base)
        for side, row in enumerate(rows):
            row["contribution_id"] = f"00000000-0000-0000-0000-{index*2+side+1:012d}"
            row["messages"][0]["content"] = f"Fixture prompt {index}"
            row["messages"][1]["content"] = "A" if side == 0 else "B"
        export["records"].extend(rows)
        review["pairs"].append({"chosen_contribution_id": rows[0]["contribution_id"],
            "rejected_contribution_id": rows[1]["contribution_id"], "preference_confirmed": True})
    export["item_count"] = len(export["records"])
    export["manifest_sha256"] = canonical_hash({"dataset_version": export["dataset_version"], "records": export["records"]})
    review["export_manifest_sha256"] = export["manifest_sha256"]
    paths = [tmp_path / "export.json", tmp_path / "review.json"]
    for path, data in zip(paths, (export, review)):
        path.write_text(json.dumps(data), encoding="utf-8")
    return parent, *paths


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_preference_candidate_and_exact_resume(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    parent, export, review = inputs(tmp_path)
    original = parent.read_bytes()
    options = dict(device=device, steps=6, validation_fraction=.5, learning_rate=.003)
    observed = []
    def record(metric):
        observed.append(dict(metric))
        metric["dpo_loss"] = -999  # Display consumers cannot mutate saved history.
    full = train_preferences(parent, export, review, tmp_path / "full", on_metric=record, **options)
    train_preferences(parent, export, review, tmp_path / "partial", stop_after=3, **options)
    resumed = train_preferences(parent, export, review, tmp_path / "resumed",
        resume=tmp_path / "partial/candidate.pt", **options)
    left, _, state = load_checkpoint(tmp_path / "full/candidate.pt")
    assert observed == state["metrics"]
    assert [metric["step"] for metric in observed] == list(range(1, 7))
    right, _, _ = load_checkpoint(tmp_path / "resumed/candidate.pt")
    assert all(torch.equal(value, right.state_dict()[name]) for name, value in left.state_dict().items())
    assert full["candidate"] == resumed["candidate"]
    assert state["data_visibility"] == "reviewed-training-only" and full["promoted"] is False
    assert full["candidate"]["mean_margin"] > full["baseline"]["mean_margin"]
    assert parent.read_bytes() == original
    assert set(state["training_split_groups"]).isdisjoint(state["validation_split_groups"])
    with pytest.raises(ValueError, match="identity"):
        train_preferences(parent, export, review, tmp_path / "invalid",
            resume=tmp_path / "partial/candidate.pt", **{**options, "beta": .2})
    assert not (tmp_path / "invalid").exists()


def test_cancelled_preference_update_is_saved_as_restricted_candidate(tmp_path, monkeypatch):
    parent, export, review = inputs(tmp_path)
    cancel = threading.Event()
    original_step = torch.optim.AdamW.step
    def stop_after_update(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        cancel.set()
        return result
    monkeypatch.setattr(torch.optim.AdamW, "step", stop_after_update)
    report = train_preferences(parent, export, review, tmp_path / "cancelled", steps=6,
                               validation_fraction=.5, cancel=cancel)
    assert report["step"] == 1 and report["cancelled"] and report["candidate"] is None
    _, _, state = load_checkpoint(tmp_path / "cancelled/candidate.pt")
    assert state["step"] == 1 and state["data_visibility"] == "reviewed-training-only"


def test_parent_contamination_rejected_before_output(tmp_path):
    from ai_home.foundation.checkpoint import atomic_save
    from ai_home.foundation.preference_data import load_preference_pairs, split_preference_pairs
    parent, export, review = inputs(tmp_path)
    _, validation = split_preference_pairs(load_preference_pairs(export, review), .5)
    _, _, state = load_checkpoint(parent)
    state["training_split_groups"].append(validation[0]["source_split_groups"][0])
    atomic_save(state, parent)
    with pytest.raises(ValueError, match="overlaps"):
        train_preferences(parent, export, review, tmp_path / "contaminated", validation_fraction=.5)
    assert not (tmp_path / "contaminated").exists()


def test_periodic_checkpoint_recovers_failed_update_exactly(tmp_path, monkeypatch):
    parent, export, review = inputs(tmp_path)
    options = dict(steps=6, checkpoint_every=2, validation_fraction=.5)
    train_preferences(parent, export, review, tmp_path / "full", **options)
    original_step, calls = torch.optim.AdamW.step, []
    def fail_third(optimizer, *args, **kwargs):
        calls.append(True)
        if len(calls) == 3:
            raise RuntimeError("controlled interrupted update")
        return original_step(optimizer, *args, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(torch.optim.AdamW, "step", fail_third)
        with pytest.raises(RuntimeError, match="controlled"):
            train_preferences(parent, export, review, tmp_path / "failed", **options)
    periodic = tmp_path / "failed/step-2.pt"
    assert periodic.exists() and not (tmp_path / "failed/candidate.pt").exists()
    train_preferences(parent, export, review, tmp_path / "recovered", resume=periodic, **options)
    expected, _, _ = load_checkpoint(tmp_path / "full/candidate.pt")
    actual, _, _ = load_checkpoint(tmp_path / "recovered/candidate.pt")
    assert all(torch.equal(value, actual.state_dict()[name]) for name, value in expected.state_dict().items())


def test_resume_rejects_changed_architecture_with_same_metadata(tmp_path):
    from ai_home.foundation.checkpoint import atomic_save
    parent, export, review = inputs(tmp_path)
    options = dict(steps=4, validation_fraction=.5)
    train_preferences(parent, export, review, tmp_path / "partial", stop_after=2, **options)
    _, _, state = load_checkpoint(tmp_path / "partial/candidate.pt")
    # Context changes do not alter tensor shapes, so the general loader accepts it.
    state["config"]["context"] += 1
    altered = tmp_path / "changed-context.pt"
    atomic_save(state, altered)
    load_checkpoint(altered)
    with pytest.raises(ValueError, match="identity"):
        train_preferences(parent, export, review, tmp_path / "invalid-architecture", resume=altered, **options)
    assert not (tmp_path / "invalid-architecture").exists()


def test_preference_cli_stops_and_resumes_real_training(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    parent, export, review = inputs(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "ai_home.foundation.preference_training",
        "--parent", str(parent), "--export", str(export), "--review", str(review),
        "--steps", "3", "--checkpoint-every", "1", "--validation-fraction", ".5"]
    first = tmp_path / "cli-first"
    result = subprocess.run(command + ["--output-dir", str(first), "--stop-after", "2"],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["step"] == 2 and (first / "step-1.pt").exists()
    result = subprocess.run(command + ["--output-dir", str(tmp_path / "cli-resumed"),
        "--resume", str(first / "candidate.pt")], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["step"] == 3 and report["training_complete"] and report["promoted"] is False
