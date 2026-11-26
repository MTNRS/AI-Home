"""Measure learned-token efficiency on exact retained training/validation documents."""
import argparse
import json
from pathlib import Path

from ai_home.foundation.checkpoint import load_checkpoint
from ai_home.foundation.data import load_records
from ai_home.foundation.tokenizer import ByteBPE
from ai_home.immutable_artifacts import publish_bytes

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--training", type=Path, required=True)
parser.add_argument("--validation", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
model, tokenizer, state = load_checkpoint(args.checkpoint, max_bytes=100_000_000)
baseline = ByteBPE()
partitions = {}
for name, path in (("training", args.training), ("validation", args.validation)):
    records, digest = load_records(path)
    declared = state.get(f"{name}_document_hashes")
    if declared is None or {row["sha256"] for row in records} != set(declared):
        raise ValueError(f"{name} dataset differs from checkpoint document lineage")
    documents = []
    for row in records:
        text = row["text"]
        learned = tokenizer.encode(text)
        raw = baseline.encode(text)
        if tokenizer.decode(learned) != text or baseline.decode(raw) != text:
            raise ValueError("Tokenizer round-trip mismatch")
        if len(raw) != len(text.encode("utf-8")):
            raise ValueError("Byte baseline mismatch")
        documents.append({"sha256": row["sha256"], "utf8_bytes": len(raw), "tokens": len(learned),
                          "language": row["language"]})
    byte_count = sum(row["utf8_bytes"] for row in documents)
    token_count = sum(row["tokens"] for row in documents)
    partitions[name] = {"dataset_sha256": digest, "documents": documents,
        "utf8_bytes": byte_count, "tokens": token_count,
        "bytes_per_token": byte_count / token_count,
        "token_reduction_fraction": 1 - token_count / byte_count}
report = {"format": "ai-home-tokenizer-efficiency-v1", "checkpoint_sha256": model.checkpoint_sha256,
    "tokenizer_sha256": tokenizer.fingerprint, "vocab_size": tokenizer.vocab_size,
    "baseline_vocab_size": baseline.vocab_size, "partitions": partitions,
    "scope": "Raw normalized text only; no BOS/EOS/PAD or conversation wrappers. No throughput or language-quality claim."}
publish_bytes(args.output, json.dumps(report, indent=2).encode("utf-8"))
print(json.dumps({name: {key: value for key, value in row.items() if key != "documents"}
                  for name, row in partitions.items()}, indent=2))
