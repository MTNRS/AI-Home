"""Experimental low-rank q/v updates for the own Foundation architecture."""
import copy
import math
import io
import re

import torch
from torch import nn
from torch.nn import functional as F

from .model import FoundationModel
from .checkpoint import FORMAT, load_checkpoint, read_checkpoint_payload
from .data import prepare_records, canonical_hash
from ai_home.immutable_artifacts import publish_bytes

ADAPTER_FORMAT = "ai-home-low-rank-adapter-v1"


class LowRankLinear(nn.Module):
    def __init__(self, base, rank, alpha, generator):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        initial = torch.randn(rank, base.in_features, generator=generator) * .02
        self.a = nn.Parameter(initial.to(device=base.weight.device, dtype=base.weight.dtype))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale

    def merged(self):
        result = copy.deepcopy(self.base)
        with torch.no_grad():
            result.weight.add_((self.b @ self.a) * self.scale)
        if not torch.isfinite(result.weight).all():
            raise ValueError("Non-finite merged adapter weights")
        return result


def attach_adapters(model, *, rank=4, alpha=4.0, seed=42):
    """Return an independent frozen-base copy; only q/v factors train."""
    if (not isinstance(model, FoundationModel) or type(rank) is not int or rank < 1
            or type(alpha) not in (float, int) or not math.isfinite(alpha) or alpha <= 0
            or type(seed) is not int):
        raise ValueError("Invalid adapter configuration")
    targets = [getattr(block.attention, name) for block in model.blocks for name in ("q", "v")]
    if any(not isinstance(layer, nn.Linear) or rank > min(layer.in_features, layer.out_features)
           for layer in targets):
        raise ValueError("Adapter rank exceeds projection dimensions or model already adapted")
    result = copy.deepcopy(model).requires_grad_(False)
    if hasattr(result, "checkpoint_sha256"):
        del result.checkpoint_sha256
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for block in result.blocks:
        for name in ("q", "v"):
            setattr(block.attention, name, LowRankLinear(getattr(block.attention, name), rank, alpha, generator))
    return result


def merge_adapters(model):
    """Return an ordinary Foundation model copy for inference, without publishing."""
    if not isinstance(model, FoundationModel) or not all(
            isinstance(getattr(block.attention, name), LowRankLinear)
            for block in model.blocks for name in ("q", "v")):
        raise ValueError("Expected adapted Foundation model")
    result = copy.deepcopy(model)
    for block in result.blocks:
        for name in ("q", "v"):
            setattr(block.attention, name, getattr(block.attention, name).merged())
    # A merged candidate no longer has the hash of the source checkpoint.
    if hasattr(result, "checkpoint_sha256"):
        del result.checkpoint_sha256
    return result


def adapter_lineage(parent_payload, training_records, validation_records):
    """Validate public records and retain cumulative parent split fingerprints."""
    training = prepare_records(training_records)
    validation = prepare_records(validation_records)
    result = {"dataset_sha256": canonical_hash({"training": training, "validation": validation})}
    for split, records in (("training", training), ("validation", validation)):
        for suffix, field in (("document_hashes", "sha256"), ("split_groups", "split_group_sha256")):
            key = split + "_" + suffix
            if key not in parent_payload:
                raise ValueError("Adapter parent requires full split lineage")
            result[key] = sorted(set(parent_payload[key]) | {row[field] for row in records if field in row})
    validate_adapter_lineage(result)
    return result


def validate_adapter_lineage(lineage):
    keys = {split + "_" + suffix for split in ("training", "validation")
            for suffix in ("document_hashes", "split_groups")}
    def digest(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    if not isinstance(lineage, dict) or set(lineage) != keys | {"dataset_sha256"} or not digest(lineage["dataset_sha256"]):
        raise ValueError("Invalid adapter lineage")
    for key in keys:
        values = lineage[key]
        if (not isinstance(values, list) or len(values) > 100000 or any(not digest(v) for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("Invalid adapter lineage fingerprints")
    if any(set(lineage["training_" + suffix]) & set(lineage["validation_" + suffix])
           for suffix in ("document_hashes", "split_groups")):
        raise ValueError("Adapter training/validation contamination")


def save_adapter(model, parent_path, output_path, *, training_records=None, validation_records=None):
    """Save only factors after checking every frozen tensor against the parent."""
    parent, tokenizer, parent_payload = load_checkpoint(parent_path, max_bytes=100_000_000)
    if (training_records is None) != (validation_records is None):
        raise ValueError("Both adapter partitions are required")
    lineage = None if training_records is None else adapter_lineage(parent_payload, training_records, validation_records)
    if not isinstance(model, FoundationModel) or model.config != parent.config:
        raise ValueError("Adapter parent architecture mismatch")
    modules = {name: layer for name, layer in model.named_modules() if isinstance(layer, LowRankLinear)}
    expected_names = {f"blocks.{i}.attention.{name}" for i in range(len(parent.blocks)) for name in ("q", "v")}
    if set(modules) != expected_names:
        raise ValueError("Invalid adapter targets")
    settings = {(layer.a.shape[0], layer.scale * layer.a.shape[0]) for layer in modules.values()}
    if len(settings) != 1:
        raise ValueError("Inconsistent adapter settings")
    rank, alpha = settings.pop()
    factors = {name + "." + side: getattr(layer, side).detach().cpu().clone()
               for name, layer in modules.items() for side in ("a", "b")}
    frozen = {key.replace(".base.weight", ".weight"): value.detach().cpu()
              for key, value in model.state_dict().items() if key not in factors}
    if set(frozen) != set(parent.state_dict()) or any(
            not torch.equal(value, parent.state_dict()[key]) for key, value in frozen.items()):
        raise ValueError("Adapter frozen base differs from parent")
    if any(not torch.isfinite(value).all() for value in factors.values()):
        raise ValueError("Non-finite adapter factors")
    payload = {"format": ADAPTER_FORMAT, "parent_sha256": parent.checkpoint_sha256,
               "tokenizer_sha256": tokenizer.fingerprint, "rank": rank, "alpha": alpha,
               "data_visibility": "reviewed-training-only", "status": "candidate", "factors": factors}
    if lineage is not None:
        payload["lineage"] = lineage
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    if buffer.tell() > 16_000_000:
        raise ValueError("Adapter exceeds file budget")
    publish_bytes(output_path, buffer.getvalue())


def load_adapter(parent_path, adapter_path, *, device="cpu"):
    payload, digest = read_checkpoint_payload(adapter_path, max_bytes=16_000_000)
    required = {"format", "parent_sha256", "tokenizer_sha256", "rank", "alpha",
                "data_visibility", "status", "factors"}
    if (not isinstance(payload, dict) or set(payload) not in (required, required | {"lineage"})
            or payload["format"] != ADAPTER_FORMAT or payload["status"] != "candidate"
            or payload["data_visibility"] != "reviewed-training-only"):
        raise ValueError("Invalid restricted adapter artifact")
    parent, tokenizer, parent_payload = load_checkpoint(parent_path, max_bytes=100_000_000)
    if payload["parent_sha256"] != parent.checkpoint_sha256 or payload["tokenizer_sha256"] != tokenizer.fingerprint:
        raise ValueError("Adapter parent identity mismatch")
    if "lineage" in payload:
        validate_adapter_lineage(payload["lineage"])
        for key in ("training_document_hashes", "validation_document_hashes", "training_split_groups", "validation_split_groups"):
            if key not in parent_payload or not set(parent_payload[key]).issubset(payload["lineage"][key]):
                raise ValueError("Adapter lineage omits parent history")
    model = attach_adapters(parent, rank=payload["rank"], alpha=payload["alpha"])
    expected = {key: value for key, value in model.named_parameters() if value.requires_grad}
    factors = payload["factors"]
    if not isinstance(factors, dict) or set(factors) != set(expected) or any(
            not isinstance(factors[key], torch.Tensor) or factors[key].shape != value.shape
            or factors[key].dtype != value.dtype or not torch.isfinite(factors[key]).all()
            for key, value in expected.items()):
        raise ValueError("Malformed adapter factors")
    with torch.no_grad():
        for key, value in expected.items():
            value.copy_(factors[key])
    return model.to(device), tokenizer, {**payload, "artifact_sha256": digest}


def export_merged_adapter(parent_path, adapter_path, output_path):
    """Export an unapproved ordinary checkpoint only when lineage is available."""
    adapted, tokenizer, metadata = load_adapter(parent_path, adapter_path)
    if "lineage" not in metadata:
        raise ValueError("Merged checkpoint export requires adapter data lineage")
    merged = merge_adapters(adapted)
    payload = {"format": FORMAT, "config": merged.config.to_dict(),
        "tokenizer": tokenizer.to_dict(), "tokenizer_sha256": tokenizer.fingerprint,
        "model_state": merged.state_dict(), "parameter_count": merged.parameter_count,
        "status": "candidate", "data_visibility": "reviewed-training-only",
        "parent_checkpoint_sha256": metadata["parent_sha256"],
        "adapter_sha256": metadata["artifact_sha256"], **metadata["lineage"]}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    if buffer.tell() > 100_000_000:
        raise ValueError("Merged checkpoint exceeds file budget")
    publish_bytes(output_path, buffer.getvalue())
    return {"parent_sha256": metadata["parent_sha256"], "adapter_sha256": metadata["artifact_sha256"],
            "data_visibility": "reviewed-training-only", "promoted": False}


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Export own-model adapter factors as a restricted candidate.")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Merge validated factors into a new ordinary checkpoint")
    export.add_argument("--parent", required=True)
    export.add_argument("--adapter", required=True)
    export.add_argument("--output", required=True, help="New checkpoint path; existing files are never replaced")
    args = parser.parse_args()
    result = export_merged_adapter(args.parent, args.adapter, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
