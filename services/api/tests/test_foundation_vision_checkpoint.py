import hashlib
import struct

import pytest

torch = pytest.importorskip("torch")
from ai_home.foundation.checkpoint import atomic_save, load_checkpoint
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.tokenizer import ByteBPE
from ai_home.foundation.vision import VisionConfig, VisionPrefixModel
from ai_home.foundation.vision_checkpoint import load_vision_weights, save_vision_weights


@pytest.fixture
def saved(tmp_path):
    torch.manual_seed(8)
    tokenizer = ByteBPE.train(["imagen propia imagen propia"], vocab_size=264)
    model = VisionPrefixModel(FoundationConfig(vocab_size=tokenizer.vocab_size,
        width=16, layers=1, heads=2, kv_heads=1, hidden=32, context=12), VisionConfig(8, 4))
    pixels = torch.rand(1, 3, 8, 8)
    tokens = torch.tensor([[257, 1, 2]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    logits, _ = model(tokens, pixels)
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
    optimizer.step()
    path = tmp_path / "visual.pt"
    pixel_bytes = struct.pack("<192f", *pixels.flatten().tolist())
    provenance = {"dataset_sha256": hashlib.sha256(pixel_bytes).hexdigest(),
                  "visibility": "public", "training_allowed": True, "privacy_reviewed": True}
    save_vision_weights(model, tokenizer, path, provenance=provenance)
    return model, tokenizer, path, pixels, tokens


def test_trained_weights_roundtrip_and_text_loader_separation(saved):
    model, tokenizer, path, pixels, tokens = saved
    loaded, bpe, payload = load_vision_weights(path)
    assert bpe.fingerprint == tokenizer.fingerprint
    assert loaded.checkpoint_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.text.output.weight is loaded.text.embedding.weight
    torch.testing.assert_close(model(tokens, pixels)[0], loaded(tokens, pixels)[0], rtol=0, atol=0)
    assert payload["status"] == "candidate"
    with pytest.raises(ValueError, match="Not an AI@Home Foundation"):
        load_checkpoint(path)
    with pytest.raises(ValueError, match="file budget"):
        load_vision_weights(path, max_bytes=path.stat().st_size - 1)


@pytest.mark.parametrize("change", ["nan", "shape", "extra", "tied", "tokenizer", "permission", "format", "dtype", "budget"])
def test_invalid_visual_envelopes_rejected(saved, change):
    _, _, path, _, _ = saved
    payload = torch.load(path, weights_only=True)
    state = payload["model_state"]
    if change == "nan":
        state["projection.weight"][0, 0] = float("nan")
    elif change == "shape":
        state["projection.weight"] = state["projection.weight"][:1]
    elif change == "extra":
        state["unexpected"] = torch.ones(1)
    elif change == "tied":
        state["text.output.weight"][0, 0] += 1
    elif change == "tokenizer":
        payload["tokenizer_sha256"] = "0" * 64
    elif change == "permission":
        payload["provenance"]["training_allowed"] = False
    elif change == "format":
        payload["format"] = "ai-home-foundation-v1"
    elif change == "dtype":
        state["projection.weight"] = state["projection.weight"].double()
    elif change == "budget":
        payload["config"]["hidden"] = 100_000_000
    atomic_save(payload, path)
    with pytest.raises(ValueError):
        load_vision_weights(path)


def test_save_requires_explicit_provenance(saved):
    model, tokenizer, path, _, _ = saved
    before = path.read_bytes()
    with pytest.raises(ValueError, match="provenance"):
        save_vision_weights(model, tokenizer, path, provenance={})
    assert path.read_bytes() == before
