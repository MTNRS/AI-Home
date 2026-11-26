import os
from pathlib import Path
import tempfile

import torch

from .model import FoundationConfig, FoundationModel
from .tokenizer import ByteBPE


FORMAT = "ai-home-foundation-v1"


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path, device="cpu", max_bytes=2_000_000_000):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("Checkpoint exceeds configured file budget")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Not an AI@Home Foundation checkpoint")
    tokenizer = ByteBPE.from_dict(payload["tokenizer"])
    if tokenizer.fingerprint != payload["tokenizer_sha256"]:
        raise ValueError("Tokenizer fingerprint mismatch")
    config = FoundationConfig(**payload["config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint vocabulary mismatch; implicit expansion forbidden")
    with torch.device("meta"):
        model = FoundationModel(config)
    if model.parameter_count * 4 > max_bytes:
        raise ValueError("Model exceeds configured memory budget")
    state = payload["model_state"]
    expected = model.state_dict()
    if set(state) != set(expected) or any(
        not isinstance(state[k], torch.Tensor) or state[k].shape != expected[k].shape
        or not state[k].is_floating_point() or not torch.isfinite(state[k]).all()
        for k in expected
    ):
        raise ValueError("Malformed or non-finite model state")
    model.to_empty(device="cpu")
    if config.tied_embeddings:
        model.output.weight = model.embedding.weight
        if not torch.equal(state["output.weight"], state["embedding.weight"]):
            raise ValueError("Inconsistent tied embeddings")
    model.load_state_dict(state, strict=True)
    return model.to(device), tokenizer, payload
