"""Explicit reviewed promotion of pending connector records into versioned datasets."""
import argparse
import hashlib
import json
from pathlib import Path

from ai_home.foundation.data import canonical_hash

MAX_BYTES = 8_000_000


def read_bounded(path):
    with Path(path).open("rb") as source:
        data = source.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Connector input exceeds review budget")
    return data.decode("utf-8")


def approve_records(pending_path, review_path, output_path, *, purpose="retrieval"):
    pending = [json.loads(line) for line in read_bounded(pending_path).splitlines() if line.strip()]
    review = json.loads(read_bounded(review_path))
    if review.get("format") != "ai-home-source-review-v1":
        raise ValueError("Invalid source review format")
    if not all(isinstance(review.get(key), str) and review[key].strip() for key in ("reviewer", "date")):
        raise ValueError("Review identity and date required")
    approvals = review.get("records")
    if not isinstance(approvals, list) or not approvals:
        raise ValueError("Explicit per-document review required")
    by_hash = {}
    for record in pending:
        digest = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
        if record.get("sha256") != digest:
            raise ValueError("Pending document hash mismatch")
        if digest in by_hash:
            raise ValueError("Ambiguous duplicate source document")
        by_hash[digest] = record
    selected = []
    seen = set()
    for approval in approvals:
        digest = approval.get("sha256")
        if digest not in by_hash or digest in seen:
            raise ValueError("Unknown or duplicate reviewed document")
        seen.add(digest)
        if approval.get("record_sha256") != canonical_hash(by_hash[digest]):
            raise ValueError("Reviewed source metadata hash mismatch")
        if approval.get("privacy_reviewed") is not True or approval.get("license_reviewed") is not True:
            raise ValueError("Privacy and license review required")
        for field in ("training_allowed", "retrieval_allowed"):
            if type(approval.get(field)) is not bool:
                raise ValueError("Both use permissions must be explicit booleans")
        selected.append({**by_hash[digest],
                         "training_allowed": approval["training_allowed"],
                         "retrieval_allowed": approval["retrieval_allowed"],
                         "privacy_reviewed": True, "visibility": "public",
                         "reviewer": review["reviewer"], "review_date": review["date"],
                         "review_sha256": canonical_hash(review)})
    # Retrieval validation also checks metadata for common secrets and email addresses.
    from ai_home.knowledge import _validate
    selected = _validate(selected, purpose=purpose)
    dataset = {"format_version": 1, "records": selected,
               "dataset_version": canonical_hash(selected), "review_sha256": canonical_hash(review)}
    raw = json.dumps(dataset, ensure_ascii=False, indent=2).encode("utf-8")
    if len(raw) > MAX_BYTES:
        raise ValueError("Approved dataset exceeds loader budget")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(raw)
    return dataset["dataset_version"]


def create_review_template(pending_path, output_path):
    records = [json.loads(line) for line in read_bounded(pending_path).splitlines() if line.strip()]
    review = {"format": "ai-home-source-review-v1", "reviewer": "", "date": "",
              "records": [{"sha256": record["sha256"], "record_sha256": canonical_hash(record),
                           "privacy_reviewed": False, "license_reviewed": False,
                           "training_allowed": False, "retrieval_allowed": False}
                          for record in records]}
    with Path(output_path).open("x", encoding="utf-8") as output:
        json.dump(review, output, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template")
    template.add_argument("pending", type=Path)
    template.add_argument("output", type=Path)
    approve = commands.add_parser("approve")
    approve.add_argument("pending", type=Path)
    approve.add_argument("review", type=Path)
    approve.add_argument("output", type=Path)
    approve.add_argument("--purpose", choices=["training", "retrieval"], default="retrieval")
    args = parser.parse_args()
    if args.command == "template":
        create_review_template(args.pending, args.output)
    else:
        print(approve_records(args.pending, args.review, args.output, purpose=args.purpose))


if __name__ == "__main__":
    main()
