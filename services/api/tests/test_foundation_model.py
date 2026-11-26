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
