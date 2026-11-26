"""Local FP32 adapter training on explicitly authorized public datasets."""
import json
import math
import time
import psutil
from pathlib import Path
import torch
from torch.nn import functional as F
from .adapters import attach_adapters, save_adapter, load_adapter, adapter_lineage
from .checkpoint import load_checkpoint, read_checkpoint_payload, atomic_save
from .data import load_records, split_records
from .training import select_device, _sequences, evaluate, TrainingCancelled
from .resources import available_training_bytes, estimate_training_bytes


def train_adapter(parent_path, dataset_path, output_dir, *, device="cpu", steps=20,
                  rank=4, alpha=4., learning_rate=.001, seed=42, batch_size=2,
                  checkpoint_every=10, resume=None, stop_after=None, cancel=None, on_metric=None,
                  objective="causal"):
    if objective not in ("causal", "sft"):
        raise ValueError("Invalid adapter training objective")
    if (any(type(v) is not int or v < 1 for v in (steps, batch_size, checkpoint_every))
            or steps > 10000 or batch_size > 64 or type(seed) is not int
            or type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or learning_rate <= 0 or (stop_after is not None and
                (type(stop_after) is not int or not 1 <= stop_after <= steps))):
        raise ValueError("Invalid adapter training settings")
    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise TrainingCancelled("Adapter training cancelled")
    check_cancel()
    device = select_device(device)
    records, _ = load_records(dataset_path)
    if objective == "sft" and any(not record.get("messages") for record in records):
        raise ValueError("SFT adapter training requires conversation messages in every record")
    training, validation = split_records(records)
    parent, tokenizer, parent_payload = load_checkpoint(parent_path, max_bytes=100_000_000)
    lineage = adapter_lineage(parent_payload, training, validation)
    if estimate_training_bytes(parent.config, batch_size) > available_training_bytes(device):
        raise ValueError("Insufficient memory for adapter training")
    settings = dict(device=device, steps=steps, rank=rank, alpha=alpha, learning_rate=learning_rate,
                    seed=seed, batch_size=batch_size, checkpoint_every=checkpoint_every)
    # Absence denotes the original causal objective, preserving old resume files.
    if objective != "causal":
        settings["objective"] = objective
    identity = {"parent_sha256": parent.checkpoint_sha256, "lineage": lineage,
                "settings": settings, "torch_version": str(torch.__version__)}
    sampler = torch.Generator().manual_seed(seed)
    start, history = 0, []
    if resume is not None:
        resume = Path(resume)
        if not all((resume / name).is_file() for name in ("training.pt", "adapter.pt")):
            raise ValueError("Incomplete adapter checkpoint: both training.pt and adapter.pt are required")
        state, _ = read_checkpoint_payload(resume / "training.pt", max_bytes=64_000_000)
        if not isinstance(state, dict) or state.get("identity") != identity:
            raise ValueError("Adapter resume identity mismatch")
        model, _, metadata = load_adapter(parent_path, resume / "adapter.pt", device=device)
        if (metadata["artifact_sha256"] != state.get("adapter_sha256") or metadata.get("lineage") != lineage
                or metadata["rank"] != rank or metadata["alpha"] != alpha):
            raise ValueError("Adapter resume artifact mismatch")
        start, history = state["step"], state["metrics"]
    else:
        model = attach_adapters(parent, rank=rank, alpha=alpha, seed=seed).to(device)
    final_step = steps if stop_after is None else stop_after
    if type(start) is not int or not 0 <= start < final_step:
        raise ValueError("Adapter resume must advance")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    if resume is not None:
        optimizer.load_state_dict(state["optimizer"])
        sampler.set_state(state["sampler_rng"])
    inputs, labels = _sequences(training, tokenizer, parent.config.context, objective, cancel=cancel)
    holdout = _sequences(validation, tokenizer, parent.config.context, objective, cancel=cancel)
    baseline = evaluate(parent.to(device), holdout, device, cancel=cancel)
    del parent
    check_cancel()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    step, saved = start, None
    def snapshot():
        folder = output / f"step-{step}"
        folder.mkdir()
        save_adapter(model, parent_path, folder / "adapter.pt",
                     training_records=training, validation_records=validation)
        _, digest = read_checkpoint_payload(folder / "adapter.pt", max_bytes=16_000_000)
        atomic_save({"identity": identity, "step": step, "metrics": history,
                     "adapter_sha256": digest, "optimizer": optimizer.state_dict(),
                     "sampler_rng": sampler.get_state()}, folder / "training.pt")
        return folder
    model.train()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    update_seconds, processed_tokens = 0., 0
    for step_number in range(start + 1, final_step + 1):
        if cancel is not None and cancel.is_set():
            break
        if device == "cuda":
            torch.cuda.synchronize()
        step_started = time.perf_counter()
        indices = torch.randint(len(inputs), (batch_size,), generator=sampler)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs[indices].to(device))
        loss = F.cross_entropy(logits.float().flatten(0, 1), labels[indices].to(device).flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise ValueError("Non-finite adapter update")
        step = step_number
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_started
        tokens = int((labels[indices] != -100).sum().item())
        update_seconds += elapsed
        processed_tokens += tokens
        history.append({"step": step, "loss": loss.item(), "update_seconds": elapsed,
                        "supervised_tokens": tokens, "tokens_per_second": tokens / elapsed})
        if step % checkpoint_every == 0:
            saved = snapshot()
        if on_metric is not None:
            on_metric(dict(history[-1]))
    if step == start:
        raise TrainingCancelled("Cancelled before any adapter update")
    if saved != output / f"step-{step}":
        saved = snapshot()
    cancelled = cancel is not None and cancel.is_set()
    try:
        candidate = None if cancelled else evaluate(model, holdout, device, cancel=cancel)
    except TrainingCancelled:
        cancelled, candidate = True, None
    report = {"format": "ai-home-adapter-training-v1", "identity": identity, "step": step,
              "objective": objective,
              "checkpoint": saved.name, "baseline": baseline, "candidate": candidate,
              "cancelled": cancelled, "training_complete": step == steps,
              "trainable_parameters": sum(p.numel() for p in parameters),
              "run_resources": {"new_steps": step - start, "supervised_tokens": processed_tokens,
                  "update_seconds": update_seconds, "tokens_per_second": processed_tokens / update_seconds,
                  "training_save_evaluation_seconds": time.perf_counter() - run_started,
                  "process_rss_bytes_after": psutil.Process().memory_info().rss,
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None},
              "data_visibility": "reviewed-training-only", "promoted": False}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "dataset", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--objective", choices=("causal", "sft"), default="causal")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    args = vars(parser.parse_args())
    parent, dataset, output = args.pop("parent"), args.pop("dataset"), args.pop("output_dir")
    print(json.dumps(train_adapter(parent, dataset, output, **args), indent=2))


if __name__ == "__main__":
    main()
