from dataclasses import asdict
import hashlib

import pytest
torch = pytest.importorskip("torch")

from ai_home.foundation.checkpoint import FORMAT, atomic_save
from ai_home.foundation.embeddings import FoundationEmbedder
from ai_home.foundation.model import FoundationConfig, FoundationModel
from ai_home.foundation.tokenizer import ByteBPE


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware unavailable"))])
def test_embeddings_cover_all_tokens_and_preserve_logits(tmp_path, device):
    torch.manual_seed(12)
    tokenizer = ByteBPE()
    config = FoundationConfig(vocab_size=262, width=32, layers=1, heads=4,
                              kv_heads=2, hidden=64, context=16)
    model = FoundationModel(config).eval()
    path = tmp_path / "model.pt"
    atomic_save({"format": FORMAT, "config": asdict(config), "tokenizer": tokenizer.to_dict(),
                 "tokenizer_sha256": tokenizer.fingerprint, "model_state": model.state_dict()}, path)
    embedder = FoundationEmbedder(path, device=device)
    assert embedder.identity["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    text = "Texto español con ventanas múltiples. English too."
    tokens = tokenizer.encode(text)
    states = []
    with torch.inference_mode():
        for offset in range(0, len(tokens), 16):
            batch = torch.tensor([tokens[offset:offset + 16]], device=device)
            hidden, _ = embedder.model.hidden_states(batch)
            logits, _ = embedder.model(batch)
            torch.testing.assert_close(logits, embedder.model.output(hidden))
            states.append(hidden[0])
        expected = torch.cat(states).mean(0)
        expected /= expected.norm()
    actual = torch.tensor(embedder.embed(text), device=device)
    torch.testing.assert_close(actual, expected)
    assert embedder.embed(text) == embedder.embed(text)
    assert embedder.embed(text + " changed") != embedder.embed(text)
    for invalid in ("", " ", "é" * 16001, None):
        with pytest.raises(ValueError):
            embedder.embed(invalid)
