"""Bounded public adapter-gradient execution and independent verification."""
import json
import math
import torch
from torch.nn import functional as F
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt
from .adapters import attach_adapters, LowRankLinear
from .data import canonical_hash
from .distributed import GradientPayload, TensorWire, decode_tensors, encode_tensors, load_public_task


def factor_shapes(config, rank):
    width = config["width"]
    value_width = config["kv_heads"] * (width // config["heads"])
    if rank > min(width, value_width):
        raise ValueError("Adapter rank exceeds projection dimensions")
    return {f"blocks.{index}.attention.{projection}.{side}": torch.empty(shape, device="meta")
            for index in range(config["layers"]) for projection in ("q", "v")
            for side, shape in (("a", (rank, width)),
                                ("b", (width if projection == "q" else value_width, rank)))}


class AdapterGradientPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base: GradientPayload
    rank: StrictInt = Field(ge=1, le=16)
    alpha: float = Field(gt=0, le=64, allow_inf_nan=False, strict=True)
    factors: dict[str, TensorWire] = Field(max_length=8)
    factors_public_and_redistributable: StrictBool


def validate_payload(payload):
    checked = AdapterGradientPayload.model_validate(payload)
    if not checked.factors_public_and_redistributable:
        raise ValueError("Restricted adapter factors cannot reach public nodes")
    shapes = factor_shapes(checked.base.config, checked.rank)
    factors = decode_tensors({k: v.model_dump() for k, v in checked.factors.items()}, shapes)
    return checked, factors


def execute_adapter_gradient(payload, *, device="cpu"):
    checked, factors = validate_payload(payload)
    _, base, tokenizer, records, inputs, labels = load_public_task(checked.base.model_dump(), device=device)
    model = attach_adapters(base, rank=checked.rank, alpha=checked.alpha)
    trainable = {key: value for key, value in model.named_parameters() if value.requires_grad}
    with torch.no_grad():
        for key, value in trainable.items():
            value.copy_(factors[key].to(value.device))
    tokens, total_loss = int((labels != -100).sum()), 0.
    for x, y in zip(inputs.split(2), labels.split(2)):
        loss = F.cross_entropy(model(x.to(device))[0].flatten(0, 1), y.to(device).flatten(), reduction="sum")
        (loss / tokens).backward()
        total_loss += loss.item()
    gradients = {name: value.grad for name, value in trainable.items()}
    if any(value is None or not torch.isfinite(value).all() for value in gradients.values()):
        raise ValueError("Invalid adapter gradients")
    identity = checked.model_dump()
    identity["base"].pop("shard")
    identity["tokenizer"] = tokenizer.to_dict()
    artifact = {"format": "ai-home-adapter-gradient-shard-v1", "tokens": tokens,
        "mean_loss": total_loss / tokens, "base_adapter_sha256": canonical_hash(identity),
        "dataset_sha256": canonical_hash(records), "document_hashes": [r["sha256"] for r in records],
        "gradients": encode_tensors(gradients)}
    return {"token_count": tokens, "artifact_sha256": canonical_hash(artifact),
            "artifact_json": json.dumps(artifact, sort_keys=True, separators=(",", ":"))}


def verified_adapter_gradient(payload, result):
    """Reference CPU recomputation; not an efficient decentralized verifier."""
    try:
        if (not isinstance(result, dict)
                or set(result) != {"token_count", "artifact_sha256", "artifact_json"}
                or type(result["artifact_json"]) is not str
                or len(result["artifact_json"]) > 1_000_000):
            return False
        artifact = json.loads(result["artifact_json"])
        if (not isinstance(artifact, dict)
                or set(artifact) != {"format", "tokens", "mean_loss", "base_adapter_sha256",
                                     "dataset_sha256", "document_hashes", "gradients"}
                or type(artifact["tokens"]) is not int
                or type(result["token_count"]) is not int
                or type(artifact["mean_loss"]) not in (int, float)
                or not math.isfinite(artifact["mean_loss"])):
            return False
        if canonical_hash(artifact) != result["artifact_sha256"]:
            return False
        reference = json.loads(execute_adapter_gradient(payload)["artifact_json"])
        if set(artifact) != set(reference):
            return False
        for key in ("format", "tokens", "base_adapter_sha256", "dataset_sha256", "document_hashes"):
            if artifact[key] != reference[key]:
                return False
        if (type(result["token_count"]) is not int or result["token_count"] != reference["tokens"]
                or type(artifact["mean_loss"]) not in (int, float)
                or not math.isclose(artifact["mean_loss"], reference["mean_loss"], rel_tol=1e-5, abs_tol=1e-6)):
            return False
        shapes = factor_shapes(payload["base"]["config"], payload["rank"])
        actual, expected = decode_tensors(artifact["gradients"], shapes), decode_tensors(reference["gradients"], shapes)
        return all(torch.allclose(actual[k], expected[k], atol=1e-6, rtol=1e-5) for k in actual)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return False


def aggregate_adapter_gradients(model, tokenizer, payloads, results, optimizer):
    """Verify all public shards before a token-weighted update of factors only."""
    if not 2 <= len(payloads) <= 8 or len(payloads) != len(results):
        raise ValueError("Require two to eight adapter shards")
    factors = {key: value for key, value in model.named_parameters() if value.requires_grad}
    expected_shapes = factor_shapes(model.config.to_dict(), payloads[0]["rank"])
    if set(factors) != set(expected_shapes):
        raise ValueError("Only adapter factors may be trainable")
    optimizer_parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if len(optimizer_parameters) != len(factors) or {id(p) for p in optimizer_parameters} != {id(p) for p in factors.values()}:
        raise ValueError("Optimizer must own exactly the adapter factors")
    base = {key.replace(".base.weight", ".weight"): value for key, value in model.named_parameters()
            if key not in factors}
    base_wire, factor_wire = encode_tensors(base), encode_tensors(factors)
    layers = [layer for layer in model.modules() if isinstance(layer, LowRankLinear)]
    artifacts, gradients, seen = [], [], set()
    for payload, result in zip(payloads, results):
        checked, _ = validate_payload(payload)
        if (checked.base.config != model.config.to_dict()
                or checked.base.shard.tokenizer.model_dump() != tokenizer.to_dict()
                or {k: v.model_dump() for k,v in checked.base.parameters.items()} != base_wire
                or {k: v.model_dump() for k,v in checked.factors.items()} != factor_wire
                or any(layer.a.shape[0] != checked.rank or layer.scale != checked.alpha / checked.rank for layer in layers)):
            raise ValueError("Stale or incompatible adapter shard")
        if not verified_adapter_gradient(payload, result):
            raise ValueError("Unverified adapter gradient")
        artifact = json.loads(result["artifact_json"])
        documents = set(artifact["document_hashes"])
        if not documents or seen.intersection(documents):
            raise ValueError("Overlapping adapter shards")
        seen.update(documents)
        artifacts.append(artifact)
        gradients.append(decode_tensors(artifact["gradients"], expected_shapes))
    total = sum(a["tokens"] for a in artifacts)
    optimizer.zero_grad(set_to_none=True)
    for key, parameter in factors.items():
        parameter.grad = sum(g[key] * (a["tokens"] / total) for g, a in zip(gradients, artifacts)).to(parameter.device)
    torch.nn.utils.clip_grad_norm_(list(factors.values()), 1., error_if_nonfinite=True)
    optimizer.step()
    return {"shards": len(artifacts), "tokens": total,
            "mean_loss": sum(a["mean_loss"] * a["tokens"] for a in artifacts) / total}
