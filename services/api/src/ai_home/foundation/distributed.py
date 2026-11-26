"""Small CPU gradient shards: bounded tensors, no pickle or remote code."""
import base64
from dataclasses import asdict
import json
import math

import torch
from torch.nn import functional as F
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from .data import canonical_hash, prepare_records
from .model import FoundationConfig, FoundationModel
from .tasks import TokenizePayload
from .tokenizer import ByteBPE
from .training import _sequences

GRADIENT_JOB = "foundation.gradient.v1"
MAX_PARAMETERS = 50_000


def encode_tensors(tensors):
    return {name: {"shape": list(t.shape), "data": base64.b64encode(
        bytes(t.detach().cpu().float().contiguous().view(torch.uint8).reshape(-1).tolist())).decode("ascii")}
        for name, t in tensors.items()}


class TensorWire(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shape: list[StrictInt] = Field(min_length=1, max_length=2)
    data: StrictStr = Field(max_length=270_000)


def decode_tensors(wire, expected):
    if set(wire) != set(expected):
        raise ValueError("Tensor names differ from architecture")
    decoded = {}
    total = 0
    for name, tensor in expected.items():
        item = TensorWire.model_validate(wire[name])
        if item.shape != list(tensor.shape):
            raise ValueError("Tensor shape mismatch")
        total += tensor.numel()
        if total > MAX_PARAMETERS:
            raise ValueError("Tensor budget exceeded")
        data = base64.b64decode(item.data, validate=True)
        if len(data) != tensor.numel() * 4:
            raise ValueError("Tensor byte length mismatch")
        value = torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(tensor.shape).clone()
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite tensor")
        decoded[name] = value
    return decoded


class GradientPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shard: TokenizePayload
    config: dict
    parameters: dict[str, TensorWire] = Field(max_length=30)
    torch_version: StrictStr
    weights_public_and_redistributable: StrictBool

    @model_validator(mode="after")
    def bounded_model(self):
        if not self.weights_public_and_redistributable:
            raise ValueError("Only public redistributable weights may reach volunteer nodes")
        if self.torch_version != str(torch.__version__):
            raise ValueError("Worker PyTorch version must match the coordinator")
        config = FoundationConfig(**self.config)
        self.config = asdict(config)
        if (config.width > 64 or config.layers > 2 or config.context > 128 or config.hidden > 128
            or config.vocab_size > 512 or config.gradient_checkpointing):
            raise ValueError("Architecture exceeds reference gradient task limits")
        if config.vocab_size != ByteBPE.from_dict(self.shard.tokenizer.model_dump()).vocab_size:
            raise ValueError("Tokenizer vocabulary mismatch")
        # The meta model checks tensor shapes before allocating any model weights.
        with torch.device("meta"):
            model = FoundationModel(config)
        if model.parameter_count > MAX_PARAMETERS:
            raise ValueError("Parameter budget exceeded")
        decode_tensors({k: v.model_dump() for k, v in self.parameters.items()}, dict(model.named_parameters()))
        if sum(len(r.text.encode()) for r in self.shard.records) > 4096:
            raise ValueError("Gradient shard exceeds 4096 input bytes")
        return self


def parameter_fingerprint(config, tokenizer, parameters):
    return canonical_hash({"config": config, "tokenizer": tokenizer, "parameters": parameters})


def execute_gradient(payload):
    validated = GradientPayload.model_validate(payload)
    config = FoundationConfig(**validated.config)
    # Allocate only after validation; never mutate the process-wide random seed.
    with torch.device("meta"):
        model = FoundationModel(config)
    model.to_empty(device="cpu")
    if config.tied_embeddings:
        model.output.weight = model.embedding.weight
    model.train()
    parameters = {k: v.model_dump() for k, v in validated.parameters.items()}
    decoded = decode_tensors(parameters, dict(model.named_parameters()))
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.copy_(decoded[name])
    tokenizer = ByteBPE.from_dict(validated.shard.tokenizer.model_dump())
    records = prepare_records([r.model_dump() for r in validated.shard.records])
    inputs, labels = _sequences(records, tokenizer, config.context)
    tokens = int((labels != -100).sum())
    total_loss = 0.0
    for x, y in zip(inputs.split(2), labels.split(2)):
        loss_sum = F.cross_entropy(model(x)[0].flatten(0, 1), y.flatten(), reduction="sum")
        (loss_sum / tokens).backward()
        total_loss += loss_sum.item()
    gradients = {name: p.grad for name, p in model.named_parameters()}
    if any(value is None or not torch.isfinite(value).all() for value in gradients.values()):
        raise ValueError("Non-finite gradients")
    artifact = {"format": "ai-home-gradient-shard-v1", "tokens": tokens,
        "mean_loss": total_loss / tokens, "dataset_sha256": canonical_hash(records),
        "document_hashes": [r["sha256"] for r in records],
        "base_sha256": parameter_fingerprint(validated.config, tokenizer.to_dict(), parameters),
        "gradients": encode_tensors(gradients)}
    return {"token_count": tokens, "artifact_sha256": canonical_hash(artifact),
            "artifact_json": json.dumps(artifact, sort_keys=True, separators=(",", ":"))}


def verified_gradient_result(payload, result):
    """Independent recomputation; intentionally costly reference verification."""
    if set(result) != {"token_count", "artifact_sha256", "artifact_json"}:
        return False
    try:
        artifact = json.loads(result["artifact_json"])
        if canonical_hash(artifact) != result["artifact_sha256"]:
            return False
        expected = json.loads(execute_gradient(payload)["artifact_json"])
        if set(artifact) != set(expected):
            return False
        for key in ("format", "tokens", "dataset_sha256", "base_sha256", "document_hashes"):
            if artifact[key] != expected[key]:
                return False
        if result["token_count"] != expected["tokens"] or not math.isclose(
                artifact["mean_loss"], expected["mean_loss"], rel_tol=1e-5, abs_tol=1e-6):
            return False
        config = FoundationConfig(**payload["config"])
        with torch.device("meta"):
            shapes = dict(FoundationModel(config).named_parameters())
        actual = decode_tensors(artifact["gradients"], shapes)
        reference = decode_tensors(expected["gradients"], shapes)
        return all(torch.allclose(actual[k], reference[k], atol=1e-6, rtol=1e-5) for k in actual)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return False


def aggregate_gradients(model, tokenizer, results, optimizer):
    """Apply a token-weighted synchronous update after server verification."""
    current = parameter_fingerprint(asdict(model.config), tokenizer.to_dict(),
                                    encode_tensors(dict(model.named_parameters())))
    artifacts = [json.loads(result["artifact_json"]) for result in results]
    if len(artifacts) < 2 or len({a["dataset_sha256"] for a in artifacts}) != len(artifacts):
        raise ValueError("Require at least two distinct gradient shards")
    gradients, total, seen_documents = [], 0, set()
    for artifact, result in zip(artifacts, results):
        if (artifact["format"] != "ai-home-gradient-shard-v1" or artifact["base_sha256"] != current
            or canonical_hash(artifact) != result["artifact_sha256"]
            or type(artifact["tokens"]) is not int or artifact["tokens"] < 1):
            raise ValueError("Stale or invalid gradient artifact")
        documents = set(artifact["document_hashes"])
        if not documents or documents & seen_documents:
            raise ValueError("Overlapping gradient shards")
        seen_documents.update(documents)
        gradients.append(decode_tensors(artifact["gradients"], dict(model.named_parameters())))
        total += artifact["tokens"]
    optimizer.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        parameter.grad = sum(g[name] * (a["tokens"] / total) for g, a in zip(gradients, artifacts)).to(parameter.device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"tokens": total, "shards": len(artifacts),
            "mean_loss": sum(a["mean_loss"] * a["tokens"] for a in artifacts) / total}
