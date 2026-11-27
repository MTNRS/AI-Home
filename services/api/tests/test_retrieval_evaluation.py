import copy
import json
from pathlib import Path

import pytest

from ai_home.knowledge import KnowledgeIndex, build_index
from ai_home.retrieval_evaluation import evaluate


def test_labeled_retrieval_metrics_and_invalid_labels(tmp_path):
    root = Path(__file__).parent / "fixtures" / "retrieval"
    records = json.loads((root / "documents.json").read_text(encoding="utf-8"))["records"]
    suite = json.loads((root / "queries.json").read_text(encoding="utf-8"))
    path = tmp_path / "index.json"
    build_index(records, path)
    index = KnowledgeIndex(path)
    result = evaluate(index, suite, limit=1)
    hits = sum(row["hit"] for row in result["results"])
    assert result["hit_rate"] == hits / 8
    assert result["mean_recall"] == result["mean_reciprocal_rank"] == result["hit_rate"]
    assert result["index_version"] == index.version
    assert all(len(row["retrieved"]) <= 1 for row in result["results"])
    for invalid in ([], suite + suite[:1], [{**suite[0], "relevant": ["unknown"]}],
                    [{**suite[0], "query": index.chunks[0]["text"]}]):
        with pytest.raises(ValueError):
            evaluate(index, invalid)
    absent = copy.deepcopy(suite[:1])
    absent[0]["query"] = "zzzzzz"
    assert evaluate(index, absent)["hit_rate"] == 0
