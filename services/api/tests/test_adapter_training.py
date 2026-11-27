import json
import threading
import pytest
import torch
from ai_home.foundation.adapter_training import train_adapter
from ai_home.foundation.adapters import load_adapter
from ai_home.foundation.smoke import smoke_records
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.training import train, TrainingConfig


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adapter_sft_masks_prompts_and_resumes_exactly(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    records = smoke_records()
    for i, record in enumerate(records):
        prompt = f"Pregunta {i}: responde A."
        record["text"] = prompt + "\nA"
        record["messages"] = [{"role": "user", "content": prompt}, {"role": "assistant", "content": "A"}]
    dataset, parent = tmp_path / "data.json", tmp_path / "parent.pt"
    dataset.write_text(json.dumps({"format_version": 1, "records": records}), encoding="utf-8")
    train(records, parent, FoundationConfig(vocab_size=262, width=16, layers=1, heads=2,
          kv_heads=1, hidden=32, context=64), TrainingConfig(steps=1, warmup=0), device)
    options = dict(device=device, steps=2, rank=2, checkpoint_every=1, objective="sft")
    observed = []
    full = train_adapter(parent, dataset, tmp_path / "full", on_metric=observed.append, **options)
    assert full["objective"] == full["identity"]["settings"]["objective"] == "sft"
    # Byte A plus EOS per example, two examples per update; prompt tokens excluded.
    assert [row["supervised_tokens"] for row in observed] == [4, 4]
    train_adapter(parent, dataset, tmp_path / "partial", stop_after=1, **options)
    resumed = train_adapter(parent, dataset, tmp_path / "resumed", resume=tmp_path / "partial/step-1", **options)
    left, _, _ = load_adapter(parent, tmp_path / "full/step-2/adapter.pt")
    right, _, _ = load_adapter(parent, tmp_path / "resumed/step-2/adapter.pt")
    assert all(torch.equal(v, right.state_dict()[k]) for k, v in left.state_dict().items())
    assert full["candidate"] == resumed["candidate"]
    with pytest.raises(ValueError, match="identity"):
        train_adapter(parent, dataset, tmp_path / "changed", resume=tmp_path / "partial/step-1",
                      **{**options, "objective": "causal"})
    assert not (tmp_path / "changed").exists()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adapter_training_exact_resume_and_changed_settings(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    records = smoke_records()
    dataset, parent = tmp_path / "data.json", tmp_path / "parent.pt"
    dataset.write_text(json.dumps({"format_version": 1, "records": records}), encoding="utf-8")
    train(records, parent, FoundationConfig(vocab_size=262, width=16, layers=1, heads=2,
          kv_heads=1, hidden=32, context=16), TrainingConfig(steps=1, warmup=0), device)
    original = parent.read_bytes()
    options = dict(device=device, steps=4, rank=2, checkpoint_every=2)
    full = train_adapter(parent, dataset, tmp_path / "full", **options)
    train_adapter(parent, dataset, tmp_path / "partial", stop_after=2, **options)
    resumed = train_adapter(parent, dataset, tmp_path / "resumed", resume=tmp_path / "partial/step-2", **options)
    left, _, _ = load_adapter(parent, tmp_path / "full/step-4/adapter.pt")
    right, _, _ = load_adapter(parent, tmp_path / "resumed/step-4/adapter.pt")
    assert all(torch.equal(v, right.state_dict()[k]) for k,v in left.state_dict().items())
    assert full["candidate"] == resumed["candidate"]
    assert full["run_resources"]["new_steps"] == 4 and resumed["run_resources"]["new_steps"] == 2
    assert full["run_resources"]["supervised_tokens"] == 4 * 2 * 16
    assert resumed["run_resources"]["supervised_tokens"] == 2 * 2 * 16
    assert full["run_resources"]["update_seconds"] > 0
    assert full["run_resources"]["process_rss_bytes_after"] > 0
    if device == "cuda":
        assert full["run_resources"]["cuda_peak_allocated_bytes"] > 0
    else:
        assert full["run_resources"]["cuda_peak_allocated_bytes"] is None
    assert parent.read_bytes() == original and resumed["promoted"] is False
    with pytest.raises(ValueError, match="identity"):
        train_adapter(parent, dataset, tmp_path / "invalid", resume=tmp_path / "partial/step-2",
                      **{**options, "learning_rate": .02})
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("interruption", ["cancel", "failure", "checkpoint"])
def test_adapter_interruption_recovers_complete_checkpoint(tmp_path, monkeypatch, interruption):
    torch.set_num_threads(1)
    records = smoke_records()
    dataset, parent = tmp_path / "data.json", tmp_path / "parent.pt"
    dataset.write_text(json.dumps({"format_version": 1, "records": records}), encoding="utf-8")
    train(records, parent, FoundationConfig(vocab_size=262, width=16, layers=1, heads=2,
          kv_heads=1, hidden=32, context=16), TrainingConfig(steps=1, warmup=0), "cpu")
    options = dict(steps=4, rank=2, checkpoint_every=1)
    train_adapter(parent, dataset, tmp_path / "full", **options)
    cancel = threading.Event()
    original_step = torch.optim.AdamW.step
    calls = 0
    def interrupted_step(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if interruption == "failure" and calls == 2:
            raise RuntimeError("controlled optimizer failure")
        result = original_step(optimizer, *args, **kwargs)
        if interruption == "cancel":
            cancel.set()
        return result
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", interrupted_step)
        if interruption == "checkpoint":
            import ai_home.foundation.adapter_training as trainer
            original_save = trainer.atomic_save
            def interrupted_save(payload, path):
                if path.parent.name == "step-2":
                    raise RuntimeError("controlled checkpoint failure")
                return original_save(payload, path)
            patch.setattr(trainer, "atomic_save", interrupted_save)
        if interruption in ("failure", "checkpoint"):
            with pytest.raises(RuntimeError, match="controlled"):
                train_adapter(parent, dataset, tmp_path / "interrupted", **options)
            assert not (tmp_path / "interrupted/report.json").exists()
        else:
            report = train_adapter(parent, dataset, tmp_path / "interrupted", cancel=cancel, **options)
            assert report["cancelled"] and report["step"] == 1 and report["candidate"] is None
    assert (tmp_path / "interrupted/step-1/training.pt").is_file()
    if interruption == "checkpoint":
        partial = tmp_path / "interrupted/step-2"
        assert (partial / "adapter.pt").is_file() and not (partial / "training.pt").exists()
        with pytest.raises(ValueError, match="Incomplete"):
            train_adapter(parent, dataset, tmp_path / "invalid", resume=partial, **options)
        assert not (tmp_path / "invalid").exists()
        # Two individually valid files from different steps cannot form a checkpoint.
        (partial / "training.pt").write_bytes((tmp_path / "interrupted/step-1/training.pt").read_bytes())
        with pytest.raises(ValueError, match="artifact mismatch"):
            train_adapter(parent, dataset, tmp_path / "mixed", resume=partial, **options)
        assert not (tmp_path / "mixed").exists()
    else:
        assert not (tmp_path / "interrupted/step-2").exists()
    train_adapter(parent, dataset, tmp_path / "resumed", resume=tmp_path / "interrupted/step-1", **options)
    left, _, _ = load_adapter(parent, tmp_path / "full/step-4/adapter.pt")
    right, _, _ = load_adapter(parent, tmp_path / "resumed/step-4/adapter.pt")
    assert all(torch.equal(v, right.state_dict()[k]) for k,v in left.state_dict().items())
