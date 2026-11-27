"""Reproducible authored-data adapter experiment, not a general trainer."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .adapters import attach_adapters, save_adapter, load_adapter, merge_adapters
from .checkpoint import load_checkpoint
from .data import split_records, canonical_hash
from .model import FoundationConfig
from .smoke import smoke_records
from .training import train, TrainingConfig, select_device, _sequences, evaluate
from .inference import stream_text


def run(output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = select_device(device)
    torch.set_num_threads(1)
    records = smoke_records()
    (output / "dataset.json").write_text(json.dumps({"format_version": 1, "records": records}), encoding="utf-8")
    parent_path = output / "parent.pt"
    train(records, parent_path, FoundationConfig(vocab_size=288, width=32, layers=1,
          heads=4, kv_heads=2, hidden=64, context=32), TrainingConfig(steps=2, warmup=0), device)
    parent, tokenizer, _ = load_checkpoint(parent_path, device=device)
    original = {key: value.detach().cpu().clone() for key, value in parent.state_dict().items()}
    training, validation = split_records(records)
    inputs, labels = _sequences(training, tokenizer, parent.config.context)
    holdout = _sequences(validation, tokenizer, parent.config.context)
    policy = attach_adapters(parent, rank=4, alpha=4)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.01)
    sampler = torch.Generator().manual_seed(42)
    baseline = evaluate(parent, holdout, device)
    history = []
    policy.train()
    for step in range(1, 41):
        indices = torch.randint(len(inputs), (2,), generator=sampler)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = policy(inputs[indices].to(device))
        loss = F.cross_entropy(logits.flatten(0, 1), labels[indices].to(device).flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step, "loss": loss.item()})
    candidate = evaluate(policy, holdout, device)
    artifact = output / "adapter.pt"
    save_adapter(policy, parent_path, artifact, training_records=training, validation_records=validation)
    restored, _, metadata = load_adapter(parent_path, artifact, device=device)
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in policy.state_dict().items())
    assert all(torch.equal(value, parent.state_dict()[key].cpu()) for key, value in original.items())
    merged = merge_adapters(restored).eval()
    generated = "".join(stream_text(merged, tokenizer, [{"role": "user", "content": "Hola"}],
                                   max_new_tokens=16, temperature=0))
    report = {"format": "ai-home-adapter-smoke-v1", "device": device,
        "torch_version": str(torch.__version__), "parent_sha256": metadata["parent_sha256"],
        "adapter_sha256": metadata["artifact_sha256"], "adapter_file_bytes": artifact.stat().st_size,
        "base_parameters": parent.parameter_count, "trainable_parameters": sum(p.numel() for p in parameters),
        "training_sha256": canonical_hash(training), "validation_sha256": canonical_hash(validation),
        "baseline": baseline, "candidate": candidate, "steps": 40, "metrics": history,
        "generated_text_unscored": generated, "data_visibility": "reviewed-training-only",
        "promoted": False, "scope": "Authored repeated sentences; no language-quality or distributed-efficiency claim"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    report = run(args.output_dir, args.device)
    print(json.dumps({key: value for key, value in report.items() if key != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
