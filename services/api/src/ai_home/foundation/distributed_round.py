"""Recoverable local round publication from verified distributed gradient shards."""
import io
from pathlib import Path

import torch

from ai_home.immutable_artifacts import publish_bytes
from .checkpoint import load_checkpoint
from .data import canonical_hash, prepare_records
from .distributed import aggregate_gradients, verified_gradient_result


def commit_round(parent, shards, output):
    """Rebuild from parent on retry; never mutate the parent's weights or optimizer.

    Each shard is a dict with payload and result from the typed node protocol.
    The caller persists job IDs/results separately; this is the aggregation commit
    boundary, not automatic job scheduling or coordinator service recovery.
    """
    if not isinstance(shards, list) or not 2 <= len(shards) <= 8:
        raise ValueError("A recoverable round requires 2..8 shards")
    if any(not isinstance(shard, dict) or set(shard) != {"payload", "result"} for shard in shards):
        raise ValueError("Invalid round shard envelope")
    model, tokenizer, payload = load_checkpoint(parent, max_bytes=10_000_000)
    if payload.get("training_mode") != "verified-gradient-rounds-v1" or "optimizer" not in payload:
        raise ValueError("Parent must be an existing distributed round with optimizer state")
    if payload.get("data_visibility") != "public":
        raise ValueError("Distributed rounds require public lineage")
    request = canonical_hash({"parent_sha256": model.checkpoint_sha256, "shards": shards})
    output = Path(output)
    def retained():
        _, _, existing = load_checkpoint(output, max_bytes=10_000_000)
        if existing.get("round_request_sha256") != request:
            raise ValueError("Existing round belongs to a different parent or shard set")
        return existing
    if output.exists():
        return retained()
    documents = set(payload.get("training_document_hashes", []))
    validation = set(payload.get("validation_document_hashes", []))
    round_records = []
    for shard in shards:
        records = prepare_records(shard["payload"]["shard"]["records"])
        hashes = {record["sha256"] for record in records}
        if hashes & validation:
            raise ValueError("Distributed shard overlaps fixed validation data")
        if not verified_gradient_result(shard["payload"], shard["result"]):
            raise ValueError("Gradient shard failed independent verification")
        documents.update(hashes)
        round_records.extend(records)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(payload["optimizer"])
    metric = aggregate_gradients(model, tokenizer, [shard["result"] for shard in shards], optimizer)
    # Evaluation/validation results describe the parent, not these updated weights.
    result = {key: value for key, value in payload.items()
              if key not in {"evaluation", "validation", "metrics"}}
    result.update(model_state=model.state_dict(), optimizer=optimizer.state_dict(),
                  step=payload["step"] + 1, status="candidate", metrics=[metric],
                  parent_checkpoint_sha256=model.checkpoint_sha256, round_request_sha256=request,
                  parent_dataset_sha256=payload.get("dataset_sha256"),
                  dataset_sha256=canonical_hash(prepare_records(round_records)),
                  training_document_hashes=sorted(documents),
                  gradient_artifact_sha256=[shard["result"]["artifact_sha256"] for shard in shards])
    stream = io.BytesIO()
    torch.save(result, stream)
    if stream.tell() > 10_000_000:
        raise ValueError("Round checkpoint exceeds publication budget")
    try:
        publish_bytes(output, stream.getvalue())
    except FileExistsError:
        return retained()
    return result


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("shards", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.shards.open("rb") as source:
        raw = source.read(16_000_001)
    if len(raw) > 16_000_000:
        raise ValueError("Round envelope exceeds byte budget")
    result = commit_round(args.parent, json.loads(raw), args.output)
    print(json.dumps({"step": result["step"], "round_request_sha256": result["round_request_sha256"]}))


if __name__ == "__main__":
    main()
