"""Read-only, bounded checkpoint inspection for local operator interfaces."""
import argparse
import json
from pathlib import Path

from .checkpoint import load_checkpoint
from .registry import file_hash


def inspect_checkpoint(path, max_bytes=2_000_000_000):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("Checkpoint exceeds inspection budget")
    digest = file_hash(path)
    model, tokenizer, payload = load_checkpoint(path, "cpu", max_bytes=max_bytes)
    if file_hash(path) != digest:
        raise ValueError("Checkpoint changed during inspection")
    report = {
        "format": "ai-home-checkpoint-inspection-v1", "checkpoint_sha256": digest,
        "file_bytes": path.stat().st_size, "architecture": model.config.to_dict(),
        "parameter_count": model.parameter_count, "tokenizer_sha256": tokenizer.fingerprint,
        "vocabulary_size": tokenizer.vocab_size,
        "declared_parameter_count_matches": payload.get("parameter_count") == model.parameter_count,
        "inspection_device": "cpu", "training_device": payload.get("device"),
        "checkpoint_status": payload.get("status"), "step": payload.get("step"),
        "training_complete": payload.get("training_complete"), "stop_reason": payload.get("stop_reason"),
        "data_visibility": payload.get("data_visibility"),
        "embedding_delta_l2": payload.get("embedding_delta_l2"),
        "dataset_sha256": payload.get("dataset_sha256"),
        "parent_checkpoint_sha256": payload.get("parent_checkpoint_sha256"),
        "training_document_count": len(payload.get("training_document_hashes", [])),
        "validation_document_count": len(payload.get("validation_document_hashes", [])),
        "training_config": payload.get("training_config"), "validation": payload.get("validation"),
        "metrics": payload.get("metrics", []),
        "claim": "Checkpoint metadata and verified tensor structure; not capability evaluation or deployment approval",
    }
    # Reject malformed metadata rather than execute/render arbitrary tensor objects.
    raw = json.dumps(report, allow_nan=False)
    if len(raw.encode("utf-8")) > 2_000_000:
        raise ValueError("Checkpoint metadata exceeds inspection budget")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
