import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from ai_home.foundation.checkpoint import load_checkpoint, atomic_save
from ai_home.foundation.continual import run_candidate
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.training import TrainingConfig, train


def records(prefix):
    return [{"source": "authored continual fixture", "url": "local:tests", "license": "Apache-2.0",
             "date": "2026-11-26", "language": "es", "training_allowed": True,
             "privacy_reviewed": True, "visibility": "public",
             "text": f"{prefix} {i}: Aprendemos con datos autorizados. " * 8}
            for i in range(4)]


def test_continual_real_training_replay_lineage_and_regression(tmp_path):
    torch.set_num_threads(1)
    old, new, validation = records("Antiguo"), records("Nuevo"), records("Reservado")
    config = FoundationConfig(vocab_size=280, width=32, layers=2, heads=4, kv_heads=2, hidden=64, context=16)
    settings = TrainingConfig(steps=4, warmup=1, checkpoint_every=2, batch_size=2)
    parent = tmp_path / "parent.pt"
    base = train(old, parent, config, settings, "cpu", validation_records=validation)
    # Restrictions must propagate even when subsequent data are public.
    base["data_visibility"] = "reviewed-training-only"
    atomic_save(base, parent)
    original_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
    for name, data in (("new", new), ("old", old), ("validation", validation)):
        (tmp_path / f"{name}.json").write_text(json.dumps({"format_version": 1, "records": data}), encoding="utf-8")
    report = run_candidate(parent, tmp_path / "experiment", tmp_path / "validation.json",
                           tmp_path / "old.json", dataset_path=tmp_path / "new.json",
                           training=settings, device="cpu")
    candidate = load_checkpoint(tmp_path / "experiment/candidate.pt")[2]
    assert report["embedding_delta_l2"] > 0
    assert report["replay_samples_per_batch"] == 1
    assert report["regression_gate_passed"] == (report["loss_delta"] <= 0)
    assert report["candidate"]["tokens"] == report["baseline"]["tokens"]
    assert candidate["tokenizer_sha256"] == base["tokenizer_sha256"]
    assert candidate["parent_checkpoint_sha256"] == original_hash
    assert candidate["data_visibility"] == "reviewed-training-only"
    assert set(base["training_document_hashes"]) <= set(candidate["training_document_hashes"])
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == original_hash
    assert candidate["status"] == "candidate"
    # Resume with replay remains bit-exact, including frozen vocabulary and lineage.
    partial = tmp_path / "partial.pt"
    train(new, partial, training=settings, device="cpu", initial_checkpoint=parent,
          validation_records=validation, replay_records=old, stop_after=2)
    resumed = train(new, tmp_path / "resumed.pt", training=settings, device="cpu", resume=partial,
                    validation_records=validation, replay_records=old)
    for key in candidate["model_state"]:
        torch.testing.assert_close(candidate["model_state"][key], resumed["model_state"][key], atol=0, rtol=0)
    assert resumed["data_visibility"] == "reviewed-training-only"
    with pytest.raises(ValueError, match="overwrite"):
        train(new, parent, training=settings, initial_checkpoint=parent, validation_records=validation)
    with pytest.raises(ValueError, match="contamination"):
        train(validation, tmp_path / "leak.pt", training=settings, initial_checkpoint=parent,
              validation_records=records("Different"))
    with pytest.raises(ValueError, match="contamination"):
        train(new, tmp_path / "leak2.pt", training=settings, initial_checkpoint=parent, validation_records=old)
