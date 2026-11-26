"""Experimental local FP32 visual SFT, with exact-schedule optimizer resume."""

from dataclasses import asdict
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .checkpoint import atomic_save
from .resources import available_training_bytes, estimate_training_bytes
from .training import select_device
from .vision import VisionPrefixModel
from .vision_batches import VisualCaptionDataset
from .vision_checkpoint import load_vision_weights, vision_weights_payload


def train_visual(manifest, tokenizer, config, vision, training, output_dir, *,
                 device="cpu", prompt="Describe the image.", resume=None, stop_after=None, cancel=None):
    if training.precision != "fp32" or training.objective != "sft":
        raise ValueError("Visual training currently requires FP32 SFT")
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after <= training.steps):
        raise ValueError("Invalid interruption step")
    device = select_device(device)
    estimate = estimate_training_bytes(config, training.batch_size) + 20 * config.width * 3 * vision.patch_size ** 2
    if estimate > min(training.memory_budget_bytes, available_training_bytes(device)):
        raise ValueError("Visual training exceeds memory budget")
    train = VisualCaptionDataset(manifest, tokenizer, config, vision, split="train", prompt=prompt)
    validation = VisualCaptionDataset(manifest, tokenizer, config, vision, split="validation", prompt=prompt)
    binding = {"training": asdict(training), "prompt": prompt, "dataset": train.dataset_sha256,
               "device": device, "torch_version": str(torch.__version__)}
    torch.manual_seed(training.seed)
    sampler = torch.Generator().manual_seed(training.seed)
    payload = None
    if resume is None:
        model = VisionPrefixModel(config, vision).to(device)
    else:
        model, loaded_tokenizer, payload = load_vision_weights(resume, device, training.memory_budget_bytes)
        if model.text.config != config or model.vision_config != vision or loaded_tokenizer.fingerprint != tokenizer.fingerprint:
            raise ValueError("Visual resume architecture or tokenizer mismatch")
        if payload.get("training_state", {}).get("binding") != binding:
            raise ValueError("Visual resume dataset or training protocol mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate)
    start, history = 0, []
    if payload is not None:
        state = payload["training_state"]
        start = state["step"]
        if type(start) is not int or not 0 <= start < training.steps:
            raise ValueError("Invalid visual resume step")
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all() for v in values.values()):
                raise ValueError("Non-finite visual optimizer state")
        sampler.set_state(state["sampler"])
        history = list(state["history"])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    last = None

    def save(step):
        data = vision_weights_payload(model, tokenizer, provenance={"dataset_sha256": train.dataset_sha256,
            "visibility": "public", "training_allowed": True, "privacy_reviewed": True})
        data["training_state"] = {"binding": binding, "step": step, "optimizer": optimizer.state_dict(),
                                  "sampler": sampler.get_state(), "history": history}
        path = destination / f"visual-step-{step}.pt"
        atomic_save(data, path)
        return path

    for step in range(start + 1, training.steps + 1):
        if cancel is not None and cancel.is_set():
            if step > start + 1:
                last = save(step - 1)
            break
        if device == "cuda":
            torch.cuda.synchronize()
        began = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = (step - training.warmup) / max(1, training.steps - training.warmup)
        multiplier = step / max(1, training.warmup) if step <= training.warmup else .1 + .45 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = training.learning_rate * multiplier
        total_loss, tokens = 0., 0
        for _ in range(training.accumulation):
            indices = torch.randint(len(train), (training.batch_size,), generator=sampler).tolist()
            images, inputs, targets = (torch.stack(values).to(device) for values in zip(*(train[i] for i in indices)))
            logits, _ = model(inputs, images)
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite visual training loss")
            (loss / training.accumulation).backward()
            total_loss += loss.item() / training.accumulation
            tokens += targets.ne(-100).sum().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training.clip, error_if_nonfinite=True)
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        model.eval()
        validation_loss, validation_tokens = 0., 0
        with torch.inference_mode():
            for image, inputs, targets in validation:
                logits, _ = model(inputs.unsqueeze(0).to(device), image.unsqueeze(0).to(device))
                loss = F.cross_entropy(logits.float().flatten(0, 1), targets.to(device), reduction="sum")
                validation_loss += loss.item()
                validation_tokens += targets.ne(-100).sum().item()
        if not math.isfinite(validation_loss):
            raise ValueError("Non-finite visual validation loss")
        history.append({"step": step, "train_loss": total_loss,
                        "validation_loss": validation_loss / validation_tokens,
                        "supervised_tokens": tokens, "seconds": elapsed,
                        "supervised_tokens_per_second": tokens / elapsed})
        stopping = step == stop_after or (cancel is not None and cancel.is_set())
        if step % training.checkpoint_every == 0 or step == training.steps or stopping:
            last = save(step)
        if stopping:
            break
    return model, history, last
