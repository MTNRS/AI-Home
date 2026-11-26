from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .checkpoint import FORMAT, atomic_save, load_checkpoint
from .data import canonical_hash, split_records
from .model import FoundationConfig, FoundationModel
from .tokenizer import ByteBPE


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 100
    batch_size: int = 2
    accumulation: int = 1
    learning_rate: float = 0.0003
    warmup: int = 10
    clip: float = 1.0
    seed: int = 42
    checkpoint_every: int = 25
    precision: str = "fp32"
    memory_budget_bytes: int = 2_000_000_000

    def __post_init__(self):
        for key in ("steps", "batch_size", "accumulation", "checkpoint_every", "memory_budget_bytes"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"Invalid {key}")
        if not 0 <= self.warmup <= self.steps:
            raise ValueError("Warmup outside training schedule")
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("Unknown precision")
        if any(not math.isfinite(v) or v <= 0 for v in (self.learning_rate, self.clip)):
            raise ValueError("Invalid optimization settings")


def select_device(device="auto"):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu, cuda or auto")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _sequences(records, tokenizer, context):
    samples = []
    for record in records:
        ids = [257, *tokenizer.encode(record["text"]), 258]
        for start in range(0, len(ids) - context, context):
            samples.append(ids[start:start + context + 1])
    if not samples:
        raise ValueError("Split has no full sequences; reduce context or supply longer documents")
    return torch.tensor(samples, dtype=torch.long)


@torch.no_grad()
def evaluate(model, sequences, device, batch_size=2):
    was_training = model.training
    model.eval()
    total, tokens = 0.0, 0
    try:
        for batch in sequences.split(batch_size):
            batch = batch.to(device)
            logits, _ = model(batch[:, :-1])
            loss = F.cross_entropy(logits.float().flatten(0, 1), batch[:, 1:].flatten(), reduction="sum")
            total += loss.item()
            tokens += batch[:, 1:].numel()
    finally:
        model.train(was_training)
    value = total / tokens
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite validation loss")
    return {"loss": value, "tokens": tokens, "perplexity": math.exp(min(value, 80))}


def train(records, output, config=None, training=None, device="auto", resume=None,
          stop_after=None, on_step=None):
    training = training or TrainingConfig()
    device = select_device(device)
    if device == "cpu" and training.precision != "fp32":
        raise ValueError("Reference CPU trainer requires fp32")
    if training.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16")
    train_records, validation_records = split_records(records)
    dataset_hash = canonical_hash({"train": train_records, "validation": validation_records})
    torch.manual_seed(training.seed)
    sampler = torch.Generator().manual_seed(training.seed)
    start, metrics = 0, []
    payload = None
    if resume:
        model, tokenizer, payload = load_checkpoint(resume, max_bytes=training.memory_budget_bytes)
        if payload["device"] != device:
            raise ValueError("Exact resume requires the original device family")
        if payload["dataset_sha256"] != dataset_hash or payload["training_config"] != asdict(training):
            raise ValueError("Resume requires identical dataset and training configuration")
        if config is not None and model.config != config:
            raise ValueError("Resume architecture mismatch")
        config = model.config
        start, metrics = payload["step"], payload["metrics"]
    else:
        tokenizer = ByteBPE.train([r["text"] for r in train_records],
                                  vocab_size=config.vocab_size if config else 512)
        if config is None:
            config = FoundationConfig(vocab_size=tokenizer.vocab_size)
        elif config.vocab_size != tokenizer.vocab_size:
            config = FoundationConfig(**{**asdict(config), "vocab_size": tokenizer.vocab_size})
        with torch.device("meta"):
            estimate_model = FoundationModel(config)
        estimated = estimate_model.parameter_count * 20 + (
            training.batch_size * config.context * config.width * config.layers * 32
            + training.batch_size * config.heads * config.context ** 2 * config.layers * 4
        )
        budget = training.memory_budget_bytes
        if device == "cuda":
            budget = min(budget, int(torch.cuda.mem_get_info()[0] * 0.8))
        if estimated > budget:
            raise ValueError(f"Estimated training bytes {estimated} exceed budget {budget}; reduce configuration")
        model = FoundationModel(config)
    model.to(device).train()
    train_data = _sequences(train_records, tokenizer, config.context)
    validation_data = _sequences(validation_records, tokenizer, config.context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=training.precision == "fp16")
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        sampler.set_state(payload["sampler_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if device == "cuda":
            if not payload["cuda_rng"]:
                raise ValueError("Exact resume requires the original device family")
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    initial = model.embedding.weight.detach().cpu().clone()
    started = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    final_step = training.steps if stop_after is None else min(stop_after, training.steps)
    if not start < final_step:
        raise ValueError("No remaining training steps")
    dtype = torch.bfloat16 if training.precision == "bf16" else torch.float16
    output = Path(output)
    for step in range(start + 1, final_step + 1):
        progress = (step - training.warmup) / max(1, training.steps - training.warmup)
        multiplier = step / max(1, training.warmup) if step <= training.warmup else (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = training.learning_rate * multiplier
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(training.accumulation):
            indices = torch.randint(len(train_data), (training.batch_size,), generator=sampler)
            batch = train_data[indices].to(device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=training.precision != "fp32"):
                logits, _ = model(batch[:, :-1])
                loss = F.cross_entropy(logits.float().flatten(0, 1), batch[:, 1:].flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss / training.accumulation).backward()
            total_loss += loss.item() / training.accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), training.clip, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metric = {"step": step, "train_loss": total_loss, "learning_rate": optimizer.param_groups[0]["lr"],
                  "elapsed_seconds": elapsed, "tokens_per_second": (
                      (step - start) * training.batch_size * training.accumulation * config.context / elapsed),
                  "gpu_peak_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0}
        metrics.append(metric)
        if on_step:
            on_step(metric)
        if step % training.checkpoint_every == 0 or step == final_step:
            validation = evaluate(model, validation_data, device)
            metric["validation_loss"] = validation["loss"]
            payload = {"format": FORMAT, "config": asdict(config), "tokenizer": tokenizer.to_dict(),
                       "tokenizer_sha256": tokenizer.fingerprint, "model_state": model.state_dict(),
                       "training_config": asdict(training), "step": step, "optimizer": optimizer.state_dict(),
                       "scaler": scaler.state_dict(), "sampler_rng": sampler.get_state(),
                       "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                       "device": device, "dataset_sha256": dataset_hash, "metrics": metrics,
                       "validation": validation, "status": "candidate",
                       "embedding_delta_l2": (model.embedding.weight.detach().cpu() - initial).norm().item(),
                       "parameter_count": model.parameter_count}
            atomic_save(payload, output.with_name(f"{output.stem}.step-{step}.pt"))
            atomic_save(payload, output)
    return payload
