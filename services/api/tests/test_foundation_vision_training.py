from dataclasses import replace
import threading

import pytest
torch = pytest.importorskip("torch")
from test_foundation_vision_data import dataset
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.tokenizer import ByteBPE
from ai_home.foundation.training import TrainingConfig
from ai_home.foundation.vision import VisionConfig
from ai_home.foundation.vision_training import train_visual
from ai_home.foundation.vision_checkpoint import load_vision_weights


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_visual_resume_matches_continuous_and_loads_for_inference(dataset, tmp_path, device):
    manifest, _ = dataset
    config = FoundationConfig(vocab_size=262, width=16, layers=1, heads=2, kv_heads=1, hidden=32, context=16)
    training = TrainingConfig(steps=4, batch_size=1, accumulation=2, warmup=1, checkpoint_every=2, objective="sft")
    args = (manifest, ByteBPE(), config, VisionConfig(8, 4), training)
    whole, history, path = train_visual(*args, tmp_path / "whole", device=device, prompt="Q")
    _, partial_history, partial = train_visual(*args, tmp_path / "partial", device=device, prompt="Q", stop_after=2)
    resumed, resumed_history, final = train_visual(*args, tmp_path / "resume", device=device, prompt="Q", resume=partial)
    assert len(history) == len(resumed_history) == 4 and len(partial_history) == 2
    for key, value in whole.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    assert [m["train_loss"] for m in history] == [m["train_loss"] for m in resumed_history]
    loaded, _, payload = load_vision_weights(final)
    assert payload["training_state"]["step"] == 4
    assert all(m["supervised_tokens_per_second"] > 0 for m in history)
    with pytest.raises(ValueError, match="protocol"):
        train_visual(*args, tmp_path / "bad", device=device, prompt="X", resume=partial)
    with pytest.raises(ValueError, match="CUDA"):
        train_visual(*args[:4], replace(training, precision="fp16"), tmp_path / "bad", device="cpu")


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_visual_mixed_precision_resume(dataset, tmp_path, precision):
    manifest, _ = dataset
    config = FoundationConfig(vocab_size=262, width=16, layers=1, heads=2, kv_heads=1,
                              hidden=32, context=16, gradient_checkpointing=True)
    training = TrainingConfig(steps=4, batch_size=1, accumulation=2, warmup=1,
                              checkpoint_every=2, objective="sft", precision=precision)
    args = (manifest, ByteBPE(), config, VisionConfig(8, 4), training)
    whole, _, _ = train_visual(*args, tmp_path / "whole", device="cuda", prompt="Q")
    _, _, partial = train_visual(*args, tmp_path / "partial", device="cuda", prompt="Q", stop_after=2)
    resumed, _, path = train_visual(*args, tmp_path / "resume", device="cuda", prompt="Q", resume=partial)
    for key, value in whole.state_dict().items():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    _, _, payload = load_vision_weights(path)
    assert bool(payload["training_state"]["scaler"]) == (precision == "fp16")


def test_cancel_before_update_creates_no_checkpoint(dataset, tmp_path):
    manifest, _ = dataset
    cancel = threading.Event()
    cancel.set()
    _, history, path = train_visual(manifest, ByteBPE(), FoundationConfig(width=16, layers=1,
        heads=2, kv_heads=1, hidden=32, context=16, vocab_size=262), VisionConfig(8, 4),
        TrainingConfig(steps=1, warmup=0, objective="sft"), tmp_path / "cancel", prompt="Q", cancel=cancel)
    assert history == [] and path is None
