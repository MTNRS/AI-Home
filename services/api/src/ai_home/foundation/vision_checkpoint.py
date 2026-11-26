"""Versioned visual inference weights, intentionally separate from text checkpoints.

This envelope is not a resumable training checkpoint or an approval certificate.
"""

from dataclasses import asdict
import re

import torch

from .checkpoint import atomic_save, read_checkpoint_payload
from .model import FoundationConfig
from .tokenizer import ByteBPE
from .vision import VisionConfig, VisionPrefixModel


FORMAT = "ai-home-vision-weights-v1"


def _provenance(value):
    if not isinstance(value, dict) or set(value) != {"dataset_sha256", "visibility", "training_allowed", "privacy_reviewed"}:
        raise ValueError("Visual weights require explicit dataset provenance")
    if not isinstance(value["dataset_sha256"], str) or not re.fullmatch("[0-9a-f]{64}", value["dataset_sha256"]):
        raise ValueError("Invalid dataset SHA256")
    if value["visibility"] not in ("public", "restricted") or value["training_allowed"] is not True or value["privacy_reviewed"] is not True:
        raise ValueError("Dataset permissions and privacy review required")
    return dict(value)


def _validate_state(model, state):
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected) or any(
        not isinstance(state[k], torch.Tensor) or state[k].layout != torch.strided
        or state[k].shape != expected[k].shape or state[k].dtype != torch.float32
        or not torch.isfinite(state[k]).all() for k in expected
    ):
        raise ValueError("Malformed or non-finite visual model state; FP32 required")
    if model.text.config.tied_embeddings and not torch.equal(
        state["text.embedding.weight"], state["text.output.weight"]
    ):
        raise ValueError("Inconsistent tied embeddings")


def save_vision_weights(model, tokenizer, path, *, provenance):
    """Save explicit local candidate weights. Never promotes or publishes them."""
    provenance = _provenance(provenance)
    if model.text.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint vocabulary mismatch")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _validate_state(model, state)
    atomic_save({"format": FORMAT, "config": model.text.config.to_dict(),
                 "vision_config": asdict(model.vision_config),
                 "tokenizer": tokenizer.to_dict(), "tokenizer_sha256": tokenizer.fingerprint,
                 "model_state": state, "provenance": provenance, "status": "candidate"}, path)


def load_vision_weights(path, device="cpu", max_bytes=64_000_000):
    payload, digest = read_checkpoint_payload(path, max_bytes)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT or payload.get("status") != "candidate":
        raise ValueError("Not an AI@Home visual candidate weights file")
    _provenance(payload.get("provenance"))
    tokenizer = ByteBPE.from_dict(payload["tokenizer"])
    if tokenizer.fingerprint != payload["tokenizer_sha256"]:
        raise ValueError("Tokenizer fingerprint mismatch")
    config = FoundationConfig(**payload["config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint vocabulary mismatch")
    with torch.device("meta"):
        model = VisionPrefixModel(config, VisionConfig(**payload["vision_config"]))
    if sum(p.numel() for p in model.parameters()) * 4 > max_bytes:
        raise ValueError("Model exceeds configured memory budget")
    _validate_state(model, payload["model_state"])
    model.to_empty(device="cpu")
    if config.tied_embeddings:
        model.text.output.weight = model.text.embedding.weight
    model.load_state_dict(payload["model_state"], strict=True)
    model.checkpoint_sha256 = digest
    return model.to(device), tokenizer, payload
