"""Explicit append-only vocabulary migration; produces an untrained candidate."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .checkpoint import FORMAT, atomic_save, load_checkpoint
from .data import canonical_hash, load_records, prepare_records
from .model import FoundationModel


def expand_checkpoint(parent, output, records, vocab_size, seed=42, max_bytes=2_000_000_000):
    parent, output = Path(parent), Path(output)
    if output.exists() or output.resolve() == parent.resolve():
        raise ValueError("Migration requires a new output path")
    digest = hashlib.sha256(parent.read_bytes()).hexdigest() if parent.stat().st_size <= max_bytes else None
    model, tokenizer, previous = load_checkpoint(parent, max_bytes=max_bytes)
    if hashlib.sha256(parent.read_bytes()).hexdigest() != digest:
        raise ValueError("Parent changed during migration")
    if "training_document_hashes" not in previous or "validation_document_hashes" not in previous:
        raise ValueError("Parent lacks dataset lineage")
    records = prepare_records(records)
    hashes = {r["sha256"] for r in records}
    if hashes & set(previous["validation_document_hashes"]):
        raise ValueError("Tokenizer corpus includes a previous holdout")
    expanded = tokenizer.extend((r["text"] for r in records), vocab_size)
    if expanded.vocab_size == tokenizer.vocab_size:
        raise ValueError("Corpus produced no additional merges")
    config = replace(model.config, vocab_size=expanded.vocab_size)
    with torch.device("meta"):
        estimate = FoundationModel(config).parameter_count * 4
    if estimate + model.parameter_count * 4 > max_bytes:
        raise ValueError("Migration models exceed memory budget")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        candidate = FoundationModel(config)
    old = model.state_dict()
    with torch.no_grad():
        for name, tensor in candidate.state_dict().items():
            if name in ("embedding.weight", "output.weight"):
                tensor[:tokenizer.vocab_size].copy_(old[name])
            else:
                tensor.copy_(old[name])
    payload = {
        "format": FORMAT, "config": asdict(config), "tokenizer": expanded.to_dict(),
        "tokenizer_sha256": expanded.fingerprint, "model_state": candidate.state_dict(),
        "parameter_count": candidate.parameter_count, "status": "candidate",
        "training_complete": False, "stop_reason": "vocabulary_expansion_requires_training",
        "parent_checkpoint_sha256": digest,
        "data_visibility": previous.get("data_visibility", "reviewed-training-only"),
        "training_document_hashes": sorted(set(previous["training_document_hashes"]) | hashes),
        "validation_document_hashes": previous["validation_document_hashes"],
        "vocabulary_migration": {"seed": seed, "old_size": tokenizer.vocab_size,
            "new_size": expanded.vocab_size, "parent_tokenizer_sha256": tokenizer.fingerprint,
            "corpus_sha256": canonical_hash(records), "optimizer_reset_required": True},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / "candidate.pt"
        atomic_save(payload, temporary)
        load_checkpoint(temporary, max_bytes=max_bytes)
        os.link(temporary, output)  # Exclusive publication: never replace another artifact.
    return payload["vocabulary_migration"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    print(json.dumps(expand_checkpoint(args.parent, args.output, load_records(args.corpus)[0],
                                       args.vocab_size, args.seed)))


if __name__ == "__main__":
    main()
