from dataclasses import asdict
import copy

import pytest
torch = pytest.importorskip("torch")

from ai_home.foundation.checkpoint import FORMAT, atomic_save, load_checkpoint
from ai_home.foundation.distributed import aggregate_gradients, execute_gradient, encode_tensors
from ai_home.foundation.distributed_round import commit_round
from test_foundation_distributed import setup_shards


def test_round_retries_restore_optimizer_without_double_update(tmp_path, monkeypatch):
    model, tokenizer, records, payloads = setup_shards()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    aggregate_gradients(model, tokenizer, [execute_gradient(p) for p in payloads], optimizer)
    parent = tmp_path / "parent.pt"
    atomic_save({"format": FORMAT, "config": asdict(model.config), "tokenizer": tokenizer.to_dict(),
                 "tokenizer_sha256": tokenizer.fingerprint, "model_state": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "step": 1, "data_visibility": "public",
                 "training_mode": "verified-gradient-rounds-v1", "validation": {"old": True}}, parent)
    original = parent.read_bytes()
    for payload in payloads:
        payload["parameters"] = encode_tensors(dict(model.named_parameters()))
    shards = [{"payload": p, "result": execute_gradient(p)} for p in payloads]
    import ai_home.foundation.distributed_round as round_module
    publish = round_module.publish_bytes
    def interrupted(*args, **kwargs):
        raise OSError("interrupted before publication")
    monkeypatch.setattr(round_module, "publish_bytes", interrupted)
    output = tmp_path / "round-2.pt"
    with pytest.raises(OSError, match="interrupted"):
        commit_round(parent, shards, output)
    assert not output.exists() and parent.read_bytes() == original
    monkeypatch.setattr(round_module, "publish_bytes", publish)
    result = commit_round(parent, shards, output)
    committed = output.read_bytes()
    assert result["step"] == 2 and "validation" not in result
    aggregate_gradients(model, tokenizer, [s["result"] for s in shards], optimizer)
    actual, _, _ = load_checkpoint(output)
    for left, right in zip(model.parameters(), actual.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    retry = commit_round(parent, shards, output)
    assert retry["step"] == 2 and output.read_bytes() == committed and parent.read_bytes() == original
    changed = copy.deepcopy(shards)
    changed[0]["result"]["token_count"] += 1
    with pytest.raises(ValueError, match="different parent or shard"):
        commit_round(parent, changed, output)
    with pytest.raises(ValueError, match="verification"):
        commit_round(parent, changed, tmp_path / "bad.pt")
