"""Bounded synthetic-token compute benchmark, never a language-quality evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import psutil
import torch
from torch.nn import functional as F

from .model import FoundationConfig, FoundationModel
from .resources import available_training_bytes, estimate_training_bytes


def benchmark(config, *, device="cpu", batch_size=1, steps=5, warmup=2,
              budget_bytes=2_000_000_000, seed=42):
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested benchmark device unavailable")
    if (type(batch_size) is not int or not 1 <= batch_size <= 16
            or type(steps) is not int or not 1 <= steps <= 50
            or type(warmup) is not int or not 1 <= warmup <= 10
            or type(budget_bytes) is not int or budget_bytes <= 0):
        raise ValueError("Invalid benchmark bounds")
    estimated = estimate_training_bytes(config, batch_size)
    available = available_training_bytes(device)
    if estimated > min(budget_bytes, available):
        raise ValueError("Benchmark exceeds estimated memory budget")
    torch.manual_seed(seed)
    model = FoundationModel(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Random token IDs measure actual forward/backward/optimizer work only.
    tokens = torch.randint(0, config.vocab_size, (batch_size, config.context + 1), device=device)
    inputs, labels = tokens[:, :-1], tokens[:, 1:]
    initial = model.embedding.weight.detach().clone()
    losses = []
    def update():
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), labels.reshape(-1))
        if not torch.isfinite(loss):
            raise ValueError("Non-finite benchmark loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach())
    for _ in range(warmup):
        update()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    rss = process.memory_info().rss
    started = time.perf_counter()
    for _ in range(steps):
        losses.append(update())
        rss = max(rss, process.memory_info().rss)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    delta = float((model.embedding.weight.detach() - initial).abs().max())
    if delta <= 0:
        raise ValueError("Benchmark did not update model weights")
    return {"format": "ai-home-compute-benchmark-v1", "config": asdict(config),
            "parameters": model.parameter_count, "device": device, "precision": "float32",
            "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
            "torch_version": torch.__version__, "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(), "seed": seed, "batch_size": batch_size,
            "warmup_steps": warmup, "measured_steps": steps, "seconds": elapsed,
            "tokens_per_second": batch_size * config.context * steps / elapsed,
            "sampled_process_rss_bytes": rss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
            "estimated_training_bytes": estimated, "available_budget_bytes": available,
            "requested_budget_bytes": budget_bytes, "max_embedding_change": delta, "losses": losses,
            "limitations": "Synthetic random tokens, repeated batch, FP32, short run; not quality or sustained throughput. RSS is sampled, not peak."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--budget-bytes", type=int, default=2_000_000_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Benchmark output already exists")
    with args.config.open("rb") as source:
        raw = source.read(65537)
    if len(raw) > 65536:
        raise ValueError("Config exceeds byte budget")
    report = benchmark(FoundationConfig(**json.loads(raw)), device=args.device, steps=args.steps,
                       batch_size=args.batch_size, budget_bytes=args.budget_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ("parameters", "tokens_per_second", "cuda_peak_allocated_bytes")}))


if __name__ == "__main__":
    main()
