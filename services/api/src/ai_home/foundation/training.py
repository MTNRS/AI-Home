from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .checkpoint import FORMAT, atomic_save, load_checkpoint
from .data import canonical_hash, prepare_records, split_records
from .model import FoundationConfig, FoundationModel
from .tokenizer import ByteBPE
from .resources import estimate_training_bytes, available_training_bytes


class TrainingCancelled(RuntimeError):
    """Cancellation before an optimizer update; no new checkpoint produced."""


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
    objective: str = "causal"

    def __post_init__(self):
        for key in ("steps", "batch_size", "accumulation", "checkpoint_every", "memory_budget_bytes"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"Invalid {key}")
        if not 0 <= self.warmup <= self.steps:
            raise ValueError("Warmup outside training schedule")
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("Unknown precision")
        if self.objective not in ("causal", "sft"):
            raise ValueError("Unknown training objective")
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


def _sequences(records, tokenizer, context, objective="causal"):
    samples, targets = [], []
    for record in records:
        if objective == "sft":
            ids, labels = [257], [-100]
            for message in record["messages"]:
                role = message["role"]
                if role not in ("user", "assistant"):
                    raise ValueError("SFT supports user/assistant roles")
                body = [*tokenizer.encode(message["content"]), 258]
                ids.extend([260 if role == "user" else 261, *body])
                labels.extend([-100, *(body if role == "assistant" else [-100] * len(body))])
        else:
            ids = [257, *tokenizer.encode(record["text"]), 258]
            labels = ids
        for start in range(0, len(ids) - 1, context):
            x = ids[start:start + context]
            y = labels[start + 1:start + context + 1]
            x = x[:len(y)]
            if not any(t != -100 for t in y):
                continue
            samples.append(x + [256] * (context - len(x)))
            targets.append(y + [-100] * (context - len(y)))
    if not samples:
        raise ValueError("Split has no full sequences; reduce context or supply longer documents")
    return torch.tensor(samples, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


@torch.no_grad()
def evaluate(model, sequences, device, batch_size=2):
    was_training = model.training
    model.eval()
    total, tokens = 0.0, 0
    try:
        for inputs, labels in zip(sequences[0].split(batch_size), sequences[1].split(batch_size)):
            labels = labels.to(device)
            logits, _ = model(inputs.to(device))
            loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum")
            total += loss.item()
            tokens += (labels != -100).sum().item()
    finally:
        model.train(was_training)
    value = total / tokens
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite validation loss")
    return {"loss": value, "tokens": tokens, "perplexity": math.exp(min(value, 80))}


def train(records, output, config=None, training=None, device="auto", resume=None,
          stop_after=None, on_step=None, reviewed_export=None, initial_checkpoint=None,
          validation_records=None, replay_records=None, replay_fraction=0.25, cancel=None):
    if cancel is not None and cancel.is_set():
        raise TrainingCancelled("Training cancelled before initialization")
    training = training or TrainingConfig()
    if resume and initial_checkpoint:
        raise ValueError("Resume and initialization are distinct operations")
    if not 0 < replay_fraction < 1:
        raise ValueError("Replay fraction must be between zero and one")
    if replay_records and training.batch_size < 2:
        raise ValueError("Replay requires at least two samples per batch")
    device = select_device(device)
    if device == "cpu" and training.precision != "fp32":
        raise ValueError("Reference CPU trainer requires fp32")
    if training.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16")
    if reviewed_export is not None:
        from .contributions import load_reviewed_export
        if records:
            raise ValueError("Use either public records or a reviewed export")
        if training.objective != "sft":
            raise ValueError("Reviewed conversations require SFT")
        records = load_reviewed_export(reviewed_export)
    if validation_records is None:
        train_records, validation_records = split_records(records, allow_reviewed=reviewed_export is not None)
    else:
        train_records = prepare_records(records, allow_reviewed=reviewed_export is not None)
        validation_records = prepare_records(validation_records)
    replay_records = prepare_records(replay_records) if replay_records else []
    train_hashes = {r["sha256"] for r in train_records + replay_records}
    validation_hashes = {r["sha256"] for r in validation_records}
    if train_hashes & validation_hashes:
        raise ValueError("Evaluation contamination: training and validation overlap")
    dataset_hash = canonical_hash({"train": train_records, "validation": validation_records,
                                   "replay": replay_records, "replay_fraction": replay_fraction})
    torch.manual_seed(training.seed)
    sampler = torch.Generator().manual_seed(training.seed)
    start, metrics = 0, []
    payload = None
    lineage = None
    parent_sha256 = None
    if resume:
        model, tokenizer, payload = load_checkpoint(resume, max_bytes=training.memory_budget_bytes)
        if "optimizer" not in payload:
            raise ValueError("Checkpoint requires fresh training via initial_checkpoint, not resume")
        if payload["device"] != device:
            raise ValueError("Exact resume requires the original device family")
        if payload["dataset_sha256"] != dataset_hash or payload["training_config"] != asdict(training):
            raise ValueError("Resume requires identical dataset and training configuration")
        if config is not None and model.config != config:
            raise ValueError("Resume architecture mismatch")
        config = model.config
        start, metrics = payload["step"], payload["metrics"]
        lineage = payload
        parent_sha256 = payload.get("parent_checkpoint_sha256")
    elif initial_checkpoint:
        initial_path = Path(initial_checkpoint).resolve()
        output_path = Path(output).resolve()
        if initial_path == output_path or any(initial_path == output_path.with_name(
                f"{output_path.stem}.step-{s}.pt") for s in range(1, training.steps + 1)):
            raise ValueError("A candidate must not overwrite its parent")
        model, tokenizer, lineage = load_checkpoint(initial_path, max_bytes=training.memory_budget_bytes)
        parent_sha256 = hashlib.sha256(initial_path.read_bytes()).hexdigest()
        if config is not None and model.config != config:
            raise ValueError("Fine-tuning must preserve architecture and vocabulary")
        if "training_document_hashes" not in lineage or "validation_document_hashes" not in lineage:
            raise ValueError("Parent lacks dataset lineage; cannot verify evaluation isolation")
        config = model.config
    else:
        tokenizer = ByteBPE.train([r["text"] for r in train_records],
                                  vocab_size=config.vocab_size if config else 512)
        if config is None:
            config = FoundationConfig(vocab_size=tokenizer.vocab_size)
        elif config.vocab_size != tokenizer.vocab_size:
            config = FoundationConfig(**{**asdict(config), "vocab_size": tokenizer.vocab_size})
        estimated = estimate_training_bytes(config, training.batch_size)
        budget = min(training.memory_budget_bytes, available_training_bytes(device))
        if estimated > budget:
            raise ValueError(f"Estimated training bytes {estimated} exceed budget {budget}; reduce configuration")
        model = FoundationModel(config)
    if lineage is not None:
        if validation_hashes & set(lineage.get("training_document_hashes", [])):
            raise ValueError("Evaluation contamination: parent trained on validation data")
        if train_hashes & set(lineage.get("validation_document_hashes", [])):
            raise ValueError("Evaluation contamination: training includes a previous holdout")
        train_hashes.update(lineage.get("training_document_hashes", []))
        validation_hashes.update(lineage.get("validation_document_hashes", []))
    estimated = estimate_training_bytes(config, training.batch_size)
    budget = min(training.memory_budget_bytes, available_training_bytes(device))
    if estimated > budget:
        raise ValueError("Training memory estimate exceeds budget")
    model.to(device).train()
    train_data = _sequences(train_records, tokenizer, config.context, training.objective)
    validation_data = _sequences(validation_records, tokenizer, config.context, training.objective)
    replay_data = _sequences(replay_records, tokenizer, config.context, training.objective) if replay_records else None
    replay_count = max(1, min(training.batch_size - 1, round(training.batch_size * replay_fraction))) if replay_data else 0
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
    if cancel is not None and cancel.is_set():
        raise TrainingCancelled("Training cancelled before the first update")
    for step in range(start + 1, final_step + 1):
        progress = (step - training.warmup) / max(1, training.steps - training.warmup)
        multiplier = step / max(1, training.warmup) if step <= training.warmup else (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = training.learning_rate * multiplier
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(training.accumulation):
            indices = torch.randint(len(train_data[0]), (training.batch_size - replay_count,), generator=sampler)
            batch = train_data[0][indices].to(device)
            labels = train_data[1][indices].to(device)
            if replay_data is not None:
                indices = torch.randint(len(replay_data[0]), (replay_count,), generator=sampler)
                batch = torch.cat((batch, replay_data[0][indices].to(device)))
                labels = torch.cat((labels, replay_data[1][indices].to(device)))
            with torch.autocast(device_type=device, dtype=dtype, enabled=training.precision != "fp32"):
                logits, _ = model(batch)
                loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())
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
        cancelled = cancel is not None and cancel.is_set()
        if step % training.checkpoint_every == 0 or step == final_step or cancelled:
            validation = evaluate(model, validation_data, device)
            metric["validation_loss"] = validation["loss"]
            cancelled = cancel is not None and cancel.is_set()
            payload = {"format": FORMAT, "config": asdict(config), "tokenizer": tokenizer.to_dict(),
                       "tokenizer_sha256": tokenizer.fingerprint, "model_state": model.state_dict(),
                       "training_config": asdict(training), "step": step, "optimizer": optimizer.state_dict(),
                       "scaler": scaler.state_dict(), "sampler_rng": sampler.get_state(),
                       "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                       "device": device, "dataset_sha256": dataset_hash, "metrics": metrics,
                       "validation": validation, "status": "candidate",
                       "training_complete": step == training.steps,
                       "stop_reason": "completed" if step == training.steps else (
                           "cancelled" if cancelled else "step_limit" if step == final_step else "periodic_checkpoint"),
                       "data_visibility": "reviewed-training-only" if reviewed_export or (
                           lineage is not None and lineage.get("data_visibility") != "public") else "public",
                       "parent_checkpoint_sha256": parent_sha256,
                       "training_document_hashes": sorted(train_hashes),
                       "validation_document_hashes": sorted(validation_hashes),
                       "replay_samples_per_batch": replay_count,
                       "dataset_provenance": sorted({(r.get("dataset_version", ""),
                                                       r.get("manifest_sha256", "")) for r in records}),
                       "embedding_delta_l2": (model.embedding.weight.detach().cpu() - initial).norm().item(),
                       "parameter_count": model.parameter_count}
            atomic_save(payload, output.with_name(f"{output.stem}.step-{step}.pt"))
            atomic_save(payload, output)
        if cancelled:
            break
    return payload
