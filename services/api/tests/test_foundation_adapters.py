import pytest
import torch
from torch.nn import functional as F
from ai_home.foundation.model import FoundationConfig, FoundationModel
from ai_home.foundation.adapters import attach_adapters, merge_adapters


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_adapter_updates_frozen_base_and_merged_cache(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    base = FoundationModel(FoundationConfig(vocab_size=262, width=16, layers=1,
        heads=2, kv_heads=1, hidden=32, context=16)).to(device)
    original = {key: value.clone() for key, value in base.state_dict().items()}
    tokens = torch.tensor([[1, 2, 3, 4]], device=device)
    adapted = attach_adapters(base, rank=2, alpha=2)
    assert torch.equal(base(tokens)[0], adapted(tokens)[0])
    frozen = {key: value.clone() for key, value in adapted.named_parameters() if not value.requires_grad}
    trainable = [value for value in adapted.parameters() if value.requires_grad]
    assert sum(p.numel() for p in trainable) == 112
    optimizer = torch.optim.AdamW(trainable, lr=.03)
    before = adapted(tokens)[0].detach().clone()
    for _ in range(12):
        optimizer.zero_grad()
        loss = F.cross_entropy(adapted(tokens)[0].reshape(-1, 262), torch.tensor([2, 3, 4, 5], device=device))
        loss.backward()
        optimizer.step()
    assert not torch.equal(before, adapted(tokens)[0])
    assert all(torch.equal(value, dict(adapted.named_parameters())[key]) for key, value in frozen.items())
    assert all(torch.equal(value, base.state_dict()[key]) for key, value in original.items())
    merged = merge_adapters(adapted).eval()
    assert set(merged.state_dict()) == set(base.state_dict())
    assert torch.allclose(adapted(tokens)[0], merged(tokens)[0], atol=1e-5, rtol=1e-5)
    _, cache = merged(tokens[:, :3], use_cache=True)
    cached, _ = merged(tokens[:, 3:], past=cache, use_cache=True)
    assert torch.allclose(cached, merged(tokens)[0][:, -1:], atol=1e-5, rtol=1e-5)


def test_adapter_validation_preserves_input():
    model = FoundationModel(FoundationConfig(width=16, layers=1, heads=2, kv_heads=1, hidden=32))
    with pytest.raises(ValueError):
        attach_adapters(model, rank=9)
    adapted = attach_adapters(model, rank=2)
    with pytest.raises(ValueError):
        attach_adapters(adapted)
    with pytest.raises(ValueError):
        merge_adapters(model)


def test_adapter_artifact_roundtrip_identity_and_frozen_base(tmp_path):
    from ai_home.foundation.adapters import save_adapter, load_adapter
    from ai_home.foundation.checkpoint import atomic_save, load_checkpoint, FORMAT
    from ai_home.foundation.tokenizer import ByteBPE
    tokenizer = ByteBPE()
    base = FoundationModel(FoundationConfig(vocab_size=tokenizer.vocab_size, width=16,
        layers=1, heads=2, kv_heads=1, hidden=32))
    parent, artifact = tmp_path / "parent.pt", tmp_path / "adapter.pt"
    payload = {"format": FORMAT, "config": base.config.to_dict(), "tokenizer": tokenizer.to_dict(),
               "tokenizer_sha256": tokenizer.fingerprint, "model_state": base.state_dict()}
    atomic_save(payload, parent)
    loaded, _, _ = load_checkpoint(parent)
    adapted = attach_adapters(loaded, rank=2)
    assert not hasattr(adapted, "checkpoint_sha256")
    with torch.no_grad():
        adapted.blocks[0].attention.q.b.fill_(.1)
    save_adapter(adapted, parent, artifact)
    restored, _, metadata = load_adapter(parent, artifact)
    assert metadata["data_visibility"] == "reviewed-training-only"
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in adapted.state_dict().items())
    with pytest.raises(FileExistsError):
        save_adapter(adapted, parent, artifact)
    payload["extra_version"] = "different immutable checkpoint"
    other = tmp_path / "other.pt"
    atomic_save(payload, other)
    with pytest.raises(ValueError, match="identity"):
        load_adapter(other, artifact)
    with torch.no_grad():
        adapted.embedding.weight.add_(1)
    with pytest.raises(ValueError, match="frozen base"):
        save_adapter(adapted, parent, tmp_path / "invalid.pt")
    assert not (tmp_path / "invalid.pt").exists()
    from ai_home.foundation.adapters import export_merged_adapter
    with pytest.raises(ValueError, match="lineage"):
        export_merged_adapter(parent, artifact, tmp_path / "merged.pt")
    assert not (tmp_path / "merged.pt").exists()


def test_adapter_lineage_preserves_parent_and_rejects_contamination():
    from ai_home.foundation.adapters import adapter_lineage
    from ai_home.foundation.smoke import smoke_records
    from ai_home.foundation.data import split_records
    training, validation = split_records(smoke_records())
    parent = {split + "_" + suffix: [] for split in ("training", "validation")
              for suffix in ("document_hashes", "split_groups")}
    parent["training_document_hashes"] = ["a" * 64]
    lineage = adapter_lineage(parent, training, validation)
    assert "a" * 64 in lineage["training_document_hashes"]
    parent["training_document_hashes"].append(validation[0]["sha256"])
    with pytest.raises(ValueError, match="contamination"):
        adapter_lineage(parent, training, validation)
    parent["training_document_hashes"] = []
    training[0]["training_allowed"] = False
    with pytest.raises(ValueError, match="authorization"):
        adapter_lineage(parent, training, validation)


def test_merged_artifact_preserves_lineage_and_rejects_probe_contamination(tmp_path):
    from ai_home.foundation.adapters import save_adapter, export_merged_adapter
    from ai_home.foundation.checkpoint import atomic_save, load_checkpoint, FORMAT
    from ai_home.foundation.tokenizer import ByteBPE
    from ai_home.foundation.smoke import smoke_records
    from ai_home.foundation.data import split_records
    from ai_home.foundation.evaluation import evaluate_checkpoint
    tokenizer = ByteBPE()
    base = FoundationModel(FoundationConfig(vocab_size=262, width=16, layers=1, heads=2, kv_heads=1, hidden=32))
    lineage = {split + "_" + suffix: [] for split in ("training", "validation")
               for suffix in ("document_hashes", "split_groups")}
    parent, artifact, merged_path = (tmp_path / name for name in ("parent.pt", "adapter.pt", "merged.pt"))
    atomic_save({"format": FORMAT, "config": base.config.to_dict(), "tokenizer": tokenizer.to_dict(),
                 "tokenizer_sha256": tokenizer.fingerprint, "model_state": base.state_dict(), **lineage}, parent)
    training, validation = split_records(smoke_records())
    adapted = attach_adapters(base, rank=2)
    save_adapter(adapted, parent, artifact, training_records=training, validation_records=validation)
    export_merged_adapter(parent, artifact, merged_path)
    merged, _, payload = load_checkpoint(merged_path)
    assert payload["data_visibility"] == "reviewed-training-only" and "optimizer" not in payload
    assert all(torch.equal(v, merged.state_dict()[k]) for k, v in merge_adapters(adapted).state_dict().items())
    with pytest.raises(FileExistsError):
        export_merged_adapter(parent, artifact, merged_path)
    suite = {"version": "contaminated-fixture", "source": "authored", "license": "Apache-2.0",
             "cases": [{"id": "one", "category": "spanish", "prompt": training[0]["text"], "answers": ["x"]}]}
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_checkpoint(merged_path, suite, device="cpu")
