import hashlib
import pytest

torch = pytest.importorskip("torch")

from ai_home.providers import foundation_chat_result
from ai_home.foundation.training import train, TrainingConfig
from ai_home.foundation.checkpoint import load_checkpoint, atomic_save
from test_foundation_model import config, records


def test_desktop_result_binds_loaded_bytes_and_rejects_changed_next_turn(tmp_path, monkeypatch):
    import ai_home.local_llm as local
    import ai_home.foundation.inference as inference
    checkpoint = tmp_path / "own.pt"
    train(records(), checkpoint, config(), TrainingConfig(steps=1, warmup=0), "cpu")
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _, _, payload = load_checkpoint(checkpoint)
    payload["note"] = "replacement on disk during inference"
    replacement = tmp_path / "replacement.pt"
    atomic_save(payload, replacement)
    monkeypatch.setattr(local, "default_checkpoint_path", lambda: tmp_path / "unused.pt")
    original = inference.stream_text
    def replace_during_generation(*args, **kwargs):
        checkpoint.write_bytes(replacement.read_bytes())
        yield from original(*args, **kwargs, max_new_tokens=4, temperature=0)
    monkeypatch.setattr(inference, "stream_text", replace_during_generation)
    result = foundation_chat_result("own.pt", [{"role": "user", "content": "Hi"}])
    assert result.answer.strip() and result.checkpoint_sha256 == before
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() != before
    with pytest.raises(ValueError, match="checkpoint ha cambiado"):
        foundation_chat_result("own.pt", [{"role": "user", "content": "Hi"}], expected_checkpoint=before)
    with pytest.raises(ValueError, match="nombre de checkpoint"):
        foundation_chat_result("../own.pt", [{"role": "user", "content": "Hi"}])
