from dataclasses import replace
from pathlib import Path
import threading

import pytest

torch = pytest.importorskip("torch")

from ai_home.foundation.model import FoundationConfig, FoundationModel, rotary
from ai_home.foundation.training import TrainingConfig, train
from ai_home.foundation.checkpoint import load_checkpoint
from ai_home.foundation.inference import stream_tokens, stream_text


def config():
    return FoundationConfig(vocab_size=280, width=32, layers=2, heads=4, kv_heads=2,
                            hidden=64, context=16)


def records():
    common = dict(source="authored fixture", url="local:tests", license="Apache-2.0",
                  date="2026-11-26", language="es", training_allowed=True,
                  privacy_reviewed=True, visibility="public")
    return [{**common, "text": (f"Documento {i}. El modelo aprende de datos autorizados. " * 12)}
            for i in range(8)]


def test_sft_masks_prompts_and_padding_and_keeps_assistant_eos():
    from ai_home.foundation.tokenizer import ByteBPE
    from ai_home.foundation.training import _sequences
    tokenizer = ByteBPE()
    messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    inputs, labels = _sequences([{"messages": messages}], tokenizer, 16, "sft")
    assert inputs[0, :7].tolist() == [257, 260, ord("Q"), 258, 261, ord("A"), 256]
    assert labels[0, :6].tolist() == [-100, -100, -100, -100, ord("A"), 258]
    assert (labels[0, 6:] == -100).all()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware unavailable"))])
def test_causality_cache_gradients_and_rope(device):
    torch.manual_seed(7)
    model = FoundationModel(config()).to(device).eval()
    x = torch.randint(0, 256, (2, 10), device=device)
    y = x.clone()
    y[:, 5:] = 42
    full, _ = model(x)
    altered, _ = model(y)
    torch.testing.assert_close(full[:, :5], altered[:, :5])
    _, cache = model(x[:, :4], use_cache=True)
    cached, cache = model(x[:, 4:], past=cache, use_cache=True)
    torch.testing.assert_close(full[:, 4:], cached, atol=1e-6, rtol=1e-5)
    assert cache[0][0].shape == (2, 2, 10, 8)
    full.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    z = torch.randn(2, 4, 6, 8, device=device)
    torch.testing.assert_close(rotary(z, 3).square().sum(-1), z.square().sum(-1))
    torch.testing.assert_close(rotary(z, 0)[:, :, 0], z[:, :, 0])


def test_gradient_checkpointing_equivalence():
    regular = FoundationModel(config()).train()
    checked = FoundationModel(replace(config(), gradient_checkpointing=True)).train()
    checked.load_state_dict(regular.state_dict())
    x = torch.randint(0, 256, (1, 8))
    for model in (regular, checked):
        model(x)[0].square().mean().backward()
    for p, q in zip(regular.parameters(), checked.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_training_resume_and_real_generation(tmp_path):
    torch.set_num_threads(1)
    settings = TrainingConfig(steps=4, warmup=1, checkpoint_every=2, batch_size=2)
    full = train(records(), tmp_path / "full.pt", config(), settings, "cpu")
    train(records(), tmp_path / "part.pt", config(), settings, "cpu", stop_after=2)
    resumed = train(records(), tmp_path / "resumed.pt", config(), settings, "cpu", resume=tmp_path / "part.pt")
    assert full["embedding_delta_l2"] > 0
    for key in full["model_state"]:
        torch.testing.assert_close(full["model_state"][key], resumed["model_state"][key], atol=0, rtol=0)
    model, tokenizer, payload = load_checkpoint(tmp_path / "full.pt")
    assert payload["status"] == "candidate"
    generated = list(stream_tokens(model, tokenizer.encode("Hola"), max_new_tokens=8, temperature=0))
    assert all(0 <= t < tokenizer.vocab_size for t in generated)
    assert isinstance("".join(stream_text(model, tokenizer, [{"role": "user", "content": "Hola"}], max_new_tokens=8)), str)
    cancel = threading.Event()
    cancel.set()
    assert list(stream_tokens(model, [257], cancel=cancel)) == []
    with pytest.raises(ValueError, match="identical"):
        train(records(), tmp_path / "bad.pt", config(), replace(settings, learning_rate=0.1),
              "cpu", resume=tmp_path / "part.pt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_cuda_training(tmp_path, precision):
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU lacks bf16 support")
    settings = TrainingConfig(steps=4, warmup=1, precision=precision, accumulation=2)
    architecture = replace(config(), gradient_checkpointing=True)
    result = train(records(), tmp_path / "cuda.pt", architecture, settings, "cuda")
    assert result["embedding_delta_l2"] > 0
    assert result["metrics"][-1]["gpu_peak_bytes"] > 0
    train(records(), tmp_path / "partial.pt", architecture, settings, "cuda", stop_after=2)
    resumed = train(records(), tmp_path / "resumed.pt", architecture, settings, "cuda",
                    resume=tmp_path / "partial.pt")
    for key in result["model_state"]:
        torch.testing.assert_close(result["model_state"][key], resumed["model_state"][key], atol=1e-6, rtol=1e-5)
    model, tokenizer, _ = load_checkpoint(tmp_path / "resumed.pt", device="cuda")
    assert next(model.parameters()).is_cuda
    generated = list(stream_tokens(model, tokenizer.encode("Hola"), max_new_tokens=8, temperature=0))
    assert all(0 <= t < tokenizer.vocab_size for t in generated)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_controlled_training_stop_saves_exactly_resumable_optimizer_boundary(tmp_path, device):
    from ai_home.foundation.training import TrainingCancelled
    torch.set_num_threads(1)
    cancel = threading.Event()
    settings = TrainingConfig(steps=4, warmup=0, accumulation=2, checkpoint_every=4)
    complete = train(records(), tmp_path / "full.pt", config(), settings, device)
    interrupted = train(records(), tmp_path / "paused.pt", config(), settings, device,
                        cancel=cancel, on_step=lambda metric: cancel.set() if metric["step"] == 2 else None)
    assert interrupted["step"] == 2
    assert interrupted["stop_reason"] == "cancelled" and interrupted["training_complete"] is False
    assert (tmp_path / "paused.step-2.pt").is_file()
    saved = load_checkpoint(tmp_path / "paused.pt")[2]
    assert saved["stop_reason"] == "cancelled"
    resumed = train(records(), tmp_path / "resumed.pt", config(), settings, device,
                    resume=tmp_path / "paused.pt")
    assert resumed["training_complete"] is True and resumed["stop_reason"] == "completed"
    for key in complete["model_state"]:
        torch.testing.assert_close(complete["model_state"][key], resumed["model_state"][key], atol=1e-6, rtol=1e-5)
    with pytest.raises(TrainingCancelled):
        train(records(), tmp_path / "never.pt", config(), settings, device, cancel=cancel)
    assert not (tmp_path / "never.pt").exists()


@pytest.mark.parametrize("tied", [True, False])
def test_vocabulary_migration(tmp_path, tied):
    from ai_home.foundation.vocabulary import expand_checkpoint
    from ai_home.foundation.data import split_records
    parent = tmp_path / "parent.pt"
    train(records(), parent, replace(config(), tied_embeddings=tied), TrainingConfig(steps=1, warmup=0), "cpu")
    before = parent.read_bytes()
    old, tokenizer, previous = load_checkpoint(parent)
    corpus, holdout = split_records(records())
    target = tmp_path / "expanded.pt"
    report = expand_checkpoint(parent, target, corpus, tokenizer.vocab_size + 8)
    new, expanded, payload = load_checkpoint(target)
    assert parent.read_bytes() == before
    assert report["new_size"] > report["old_size"]
    assert expanded.merges[:len(tokenizer.merges)] == tokenizer.merges
    assert expanded.pieces[:tokenizer.vocab_size] == tokenizer.pieces
    text = "Documento Unicode: " + chr(0x6f22)
    assert expanded.decode(expanded.encode(text)) == text
    for name, tensor in old.state_dict().items():
        value = new.state_dict()[name]
        if name in ("embedding.weight", "output.weight"):
            value = value[:tokenizer.vocab_size]
        assert torch.equal(tensor, value)
    assert (new.output.weight is new.embedding.weight) == tied
    assert "optimizer" not in payload and not payload["training_complete"]
    assert set(payload["training_document_hashes"]) >= set(previous["training_document_hashes"])
    duplicate = tmp_path / "duplicate.pt"
    expand_checkpoint(parent, duplicate, corpus, tokenizer.vocab_size + 8)
    for name, tensor in load_checkpoint(duplicate)[0].state_dict().items():
        assert torch.equal(tensor, new.state_dict()[name])
    with pytest.raises(ValueError, match="new output"):
        expand_checkpoint(parent, target, corpus, tokenizer.vocab_size + 8)
    with pytest.raises(ValueError, match="holdout"):
        expand_checkpoint(parent, tmp_path / "bad.pt", holdout, tokenizer.vocab_size + 8)
    with pytest.raises(ValueError, match="fresh training"):
        train(records(), tmp_path / "resume.pt", training=TrainingConfig(steps=1, warmup=0), device="cpu", resume=target)
    trained = train(records(), tmp_path / "trained.pt", training=TrainingConfig(steps=1, warmup=0),
                    device="cpu", initial_checkpoint=target)
    assert trained["embedding_delta_l2"] > 0


def test_vocabulary_and_fresh_training_cli_end_to_end(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from dataclasses import asdict
    from ai_home.foundation.data import split_records
    parent = tmp_path / "parent.pt"
    settings = TrainingConfig(steps=1, warmup=0)
    train(records(), parent, config(), settings, "cpu")
    original = parent.read_bytes()
    corpus, holdout = split_records(records())
    for name, documents in (("corpus", corpus), ("holdout", holdout)):
        (tmp_path / (name + ".json")).write_text(json.dumps({"format_version": 1, "records": documents}), encoding="utf-8")
    (tmp_path / "settings.json").write_text(json.dumps(asdict(settings)), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src"), "OMP_NUM_THREADS": "1"}
    def invoke(module, *arguments):
        return subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                              env=env, capture_output=True, text=True, timeout=60)
    expanded = tmp_path / "expanded.pt"
    result = invoke("ai_home.foundation.vocabulary", parent, tmp_path / "corpus.json", expanded,
                    "--vocab-size", 290)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["new_size"] > 280
    trained = tmp_path / "trained.pt"
    args = ["--dataset", tmp_path / "corpus.json", "--validation-dataset", tmp_path / "holdout.json",
            "--training-config", tmp_path / "settings.json", "--initial-checkpoint", expanded,
            "--output", trained, "--device", "cpu", "--auto-fit"]
    result = invoke("ai_home.foundation", *args)
    assert result.returncode == 0, result.stderr
    payload = load_checkpoint(trained)[2]
    assert payload["embedding_delta_l2"] > 0 and payload["training_complete"]
    assert payload["config"] == load_checkpoint(expanded)[2]["config"]
    assert json.loads(result.stdout.splitlines()[-1])["training_complete"]
    assert parent.read_bytes() == original
    conflict = invoke("ai_home.foundation", *args, "--resume", parent)
    assert conflict.returncode == 2
    shrinking = invoke("ai_home.foundation", *args, "--allow-model-shrink")
    assert shrinking.returncode == 2


def test_local_training_job_retains_configuration_and_resumes(tmp_path):
    import json
    from dataclasses import asdict
    from ai_home.foundation.training_job import run_training
    dataset, architecture, settings = (tmp_path / name for name in ("dataset.json", "model.json", "training.json"))
    dataset.write_text(json.dumps({"format_version": 1, "records": records()}))
    architecture.write_text(json.dumps(asdict(config())))
    settings.write_text(json.dumps(asdict(TrainingConfig(steps=4, warmup=0))))
    cancellation = threading.Event()
    paused = run_training(dataset, architecture, settings, tmp_path / "paused", cancel=cancellation, disk_sequences=True,
                          on_step=lambda metric: cancellation.set() if metric["step"] == 2 else None)
    assert paused["step"] == 2 and paused["stop_reason"] == "cancelled"
    assert (tmp_path / "paused" / "configuration.json").is_file()
    assert json.loads((tmp_path / "paused/configuration.json").read_text())["disk_sequences"] is True
    assert (tmp_path / "paused/sequence-cache/train/manifest.json").is_file()
    assert json.loads((tmp_path / "paused" / "result.json").read_text())["step"] == 2
    resumed = run_training(dataset, None, settings, tmp_path / "resumed", resume=paused["checkpoint"], disk_sequences=True)
    complete = run_training(dataset, architecture, settings, tmp_path / "complete")
    assert resumed["training_complete"] and complete["training_complete"]
    left = load_checkpoint(resumed["checkpoint"])[0].state_dict()
    right = load_checkpoint(complete["checkpoint"])[0].state_dict()
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=1e-6, rtol=1e-5)
    with pytest.raises(FileExistsError):
        run_training(dataset, architecture, settings, tmp_path / "paused")


def test_checkpoint_loader_rejects_in_place_modification_during_read(tmp_path, monkeypatch):
    path = tmp_path / "changing.pt"
    train(records(), path, config(), TrainingConfig(steps=1, warmup=0), "cpu")
    original = torch.load
    def changed(source, *args, **kwargs):
        payload = original(source, *args, **kwargs)
        with path.open("r+b") as target:
            target.seek(-1, 2)
            byte = target.read(1)
            target.seek(-1, 2)
            target.write(bytes([byte[0] ^ 1]))
            target.flush()
        return payload
    monkeypatch.setattr(torch, "load", changed)
    with pytest.raises(ValueError, match="changed while loading"):
        load_checkpoint(path)
