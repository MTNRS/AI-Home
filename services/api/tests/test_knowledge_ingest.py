import bz2
import json
from pathlib import Path

import pytest

from ai_home.foundation.data import canonical_hash, load_records
from ai_home.knowledge import build_index, KnowledgeIndex
from ai_home.knowledge_ingest import approve_records
from ai_home.wikipedia_corpus import import_wikipedia_dump


def fixture(tmp_path):
    dump = tmp_path / "fixture.xml.bz2"
    dump.write_bytes(bz2.compress(b"""<mediawiki>
    <page><title>Marte</title><ns>0</ns><id>1</id><revision><id>10</id>
    <text>Marte tiene dos lunas.</text></revision></page>
    <page><title>Pan</title><ns>0</ns><id>2</id><revision><id>20</id>
    <text>La levadura fermenta la masa del pan.</text></revision></page>
    </mediawiki>"""))
    corpus = tmp_path / "corpus.txt"
    count, _, _ = import_wikipedia_dump(dump, corpus,
        source_url="https://dumps.wikimedia.org/eswiki/fixture.xml.bz2")
    assert count == 2
    pending = Path(str(corpus) + ".records.jsonl")
    records = [json.loads(line) for line in pending.read_text(encoding="utf-8").splitlines()]
    review = tmp_path / "review.json"
    data = {"format": "ai-home-source-review-v1", "reviewer": "fixture-reviewer",
            "date": "2026-11-26", "records": [
                {"sha256": record["sha256"], "record_sha256": canonical_hash(record), "privacy_reviewed": True,
                 "license_reviewed": True, "training_allowed": False,
                 "retrieval_allowed": True} for record in records]}
    review.write_text(json.dumps(data), encoding="utf-8")
    return pending, review, records, data


def test_wikipedia_review_to_index_and_training_separation(tmp_path):
    pending, review, records, _ = fixture(tmp_path)
    assert all(record["visibility"] == "pending-review" for record in records)
    with pytest.raises(ValueError):
        build_index(records, tmp_path / "unreviewed.json")
    output = tmp_path / "dataset.json"
    version = approve_records(pending, review, output)
    data, digest = load_records(output, purpose="retrieval")
    assert version == digest
    assert all(record["reviewer"] == "fixture-reviewer" for record in data)
    with pytest.raises(ValueError, match="Training authorization"):
        load_records(output)
    index = tmp_path / "index.json"
    build_index(data, index)
    hit = KnowledgeIndex(index).search("Marte")[0]
    assert hit["url"].endswith("oldid=10")
    assert hit["text"] == "Marte tiene dos lunas."
    with pytest.raises(FileExistsError):
        approve_records(pending, review, output)


@pytest.mark.parametrize("field,value", [("privacy_reviewed", False),
    ("license_reviewed", False), ("retrieval_allowed", "true"),
    ("training_allowed", 1), ("sha256", "unknown")])
def test_review_must_match_exact_document_and_explicit_permissions(tmp_path, field, value):
    pending, review, _, data = fixture(tmp_path)
    data["records"][0][field] = value
    review.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "dataset.json"
    with pytest.raises(ValueError):
        approve_records(pending, review, output)
    assert not output.exists()


def test_review_rejects_modified_source_and_duplicate_approval(tmp_path):
    pending, review, records, data = fixture(tmp_path)
    data["records"].append(data["records"][0])
    review.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        approve_records(pending, review, tmp_path / "duplicate.json")
    records[0]["text"] = "Changed after review"
    pending.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        approve_records(pending, review, tmp_path / "modified.json")


def test_review_binds_source_metadata_and_allows_training_only_when_explicit(tmp_path):
    pending, review, records, data = fixture(tmp_path)
    for approval in data["records"]:
        approval["training_allowed"] = True
        approval["retrieval_allowed"] = False
    review.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "training.json"
    approve_records(pending, review, output, purpose="training")
    prepared, _ = load_records(output)
    assert all(record["retrieval_allowed"] is False for record in prepared)
    with pytest.raises(ValueError, match="Retrieval authorization"):
        build_index(prepared, tmp_path / "forbidden.json")
    records[0]["license"] = "Changed license after review"
    pending.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata hash"):
        approve_records(pending, review, tmp_path / "changed.json", purpose="training")


def test_template_grants_no_automatic_permissions(tmp_path):
    from ai_home.knowledge_ingest import create_review_template
    pending, _, _, _ = fixture(tmp_path)
    review = tmp_path / "template.json"
    create_review_template(pending, review)
    data = json.loads(review.read_text(encoding="utf-8"))
    assert all(record["training_allowed"] is False and record["retrieval_allowed"] is False
               for record in data["records"])
    with pytest.raises(ValueError, match="identity"):
        approve_records(pending, review, tmp_path / "not-approved.json")
    with pytest.raises(FileExistsError):
        create_review_template(pending, review)
