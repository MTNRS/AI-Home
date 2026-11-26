"""Reproducible labeled retrieval evaluation, independent of generation scores."""
import json
from pathlib import Path
import time

from ai_home.foundation.data import canonical_hash
from ai_home.knowledge import KnowledgeIndex


def evaluate(index, suite, *, limit=3):
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("Invalid retrieval evaluation limit")
    if not isinstance(suite, list) or not 1 <= len(suite) <= 1000:
        raise ValueError("Expected 1..1000 labeled queries")
    documents = {chunk["document_sha256"] for chunk in index.chunks}
    seen = set()
    for probe in suite:
        if (not isinstance(probe, dict) or set(probe) != {"id", "query", "relevant"}
                or not isinstance(probe["id"], str) or not probe["id"] or probe["id"] in seen
                or not isinstance(probe["query"], str) or not probe["query"].strip()
                or len(probe["query"].encode("utf-8")) > 32000
                or not isinstance(probe["relevant"], list) or not probe["relevant"]
                or any(not isinstance(value, str) for value in probe["relevant"])
                or len(set(probe["relevant"])) != len(probe["relevant"])):
            raise ValueError("Invalid or duplicate retrieval probe")
        if not set(probe["relevant"]) <= documents:
            raise ValueError("Unknown relevant document")
        if any(probe["query"].strip() == chunk["text"].strip() for chunk in index.chunks):
            raise ValueError("Evaluation query must differ from indexed chunks")
        seen.add(probe["id"])
    rows = []
    for probe in suite:
        started = time.perf_counter()
        matches = index.search(probe["query"], limit=limit)
        elapsed = time.perf_counter() - started
        expected = set(probe["relevant"])
        retrieved = [match["document_sha256"] for match in matches]
        first = next((i for i, value in enumerate(retrieved, 1) if value in expected), None)
        rows.append({"id": probe["id"], "query": probe["query"], "relevant": probe["relevant"],
                     "retrieved": [{"chunk_id": m["id"], "document_sha256": m["document_sha256"],
                                    "score": m["score"]} for m in matches],
                     "recall": len(expected & set(retrieved)) / len(expected),
                     "reciprocal_rank": 1 / first if first else 0,
                     "hit": first is not None, "seconds": elapsed})
    return {"format": "ai-home-retrieval-evaluation-v1", "index_version": index.version,
            "embedding": index.summary()["embedding"], "suite_sha256": canonical_hash(suite),
            "limit": limit, "queries": len(rows),
            "mean_recall": sum(row["recall"] for row in rows) / len(rows),
            "mean_reciprocal_rank": sum(row["reciprocal_rank"] for row in rows) / len(rows),
            "hit_rate": sum(row["hit"] for row in rows) / len(rows), "results": rows,
            "limitations": "Authored relevance labels; not proof of broad semantic or generation quality."}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--embedding-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    with args.suite.open("rb") as source:
        raw = source.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("Retrieval suite exceeds byte budget")
    embedder = None
    if args.embedding_checkpoint:
        from ai_home.foundation.embeddings import FoundationEmbedder
        embedder = FoundationEmbedder(args.embedding_checkpoint, device=args.device)
    result = evaluate(KnowledgeIndex(args.index, embedder=embedder), json.loads(raw), limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ("queries", "mean_recall", "mean_reciprocal_rank", "hit_rate")}))


if __name__ == "__main__":
    main()
