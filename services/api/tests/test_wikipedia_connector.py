import json
import bz2

import httpx
import pytest

from ai_home.knowledge_connectors import acquire
from ai_home.knowledge_ingest import create_review_template, approve_records
from ai_home.foundation.data import canonical_hash, load_records


def spec(revision=123):
    return dict(connector="wikipedia-revision-v1", language="es", revision=revision, license="CC BY-SA 4.0")


def response(revision=123):
    return {"query": {"pages": [{"pageid": 42, "ns": 0, "title": "Article fixture",
        "revisions": [{"revid": revision, "timestamp": "2026-11-26T00:00:00Z",
            "slots": {"main": {"contentmodel": "wikitext", "content": f"Revision {revision} of [[science]]."}}}]}]}}


def test_pinned_revision_and_review_contract(tmp_path):
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=response(int(request.url.params["revids"])))
    pending = tmp_path / "wiki.jsonl"
    acquire([spec(123), spec(124)], pending, transport=httpx.MockTransport(respond))
    rows = [json.loads(line) for line in pending.read_text(encoding="utf-8").splitlines()]
    assert all(r.url.host == "es.wikipedia.org" and "authorization" not in r.headers for r in requests)
    assert all(r.url.params["rvslots"] == "main" for r in requests)
    assert all(r["visibility"] == "pending-review" and not r["training_allowed"] for r in rows)
    assert rows[0]["split_group_sha256"] == rows[1]["split_group_sha256"] == canonical_hash(
        {"family": "wikipedia-page-v1", "language": "es", "page_id": "42"})
    assert "[[" not in rows[0]["text"] and "oldid=123" in rows[0]["url"]
    review, approved = tmp_path / "review.json", tmp_path / "approved.json"
    create_review_template(pending, review)
    with pytest.raises(ValueError):
        approve_records(pending, review, approved, purpose="training")
    document = json.loads(review.read_text(encoding="utf-8"))
    document.update(reviewer="fixture reviewer", date="2026-11-26")
    for row in document["records"]:
        row.update(privacy_reviewed=True, license_reviewed=True, training_allowed=True, retrieval_allowed=False)
    review.write_text(json.dumps(document), encoding="utf-8")
    approve_records(pending, review, approved, purpose="training")
    assert len(load_records(approved)[0]) == 2


def test_network_and_dump_revisions_cannot_split_one_page_family(tmp_path):
    from ai_home.wikipedia_corpus import import_wikipedia_dump
    from ai_home.foundation.data import split_records
    dump = tmp_path / "articles.xml.bz2"
    dump.write_bytes(bz2.compress(b'<mediawiki><page><title>Article fixture</title>'
        b'<ns>0</ns><id>42</id><revision><id>122</id>'
        b'<text>Earlier article about [[science]].</text></revision></page></mediawiki>'))
    corpus = tmp_path / "corpus.txt"
    import_wikipedia_dump(dump, corpus,
        source_url="https://dumps.wikimedia.org/eswiki/20261101/eswiki-20261101-pages-articles.xml.bz2",
        language="es")
    local = json.loads((tmp_path / "corpus.txt.records.jsonl").read_text(encoding="utf-8"))
    pending = tmp_path / "network.jsonl"
    acquire([spec()], pending, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response())))
    network = json.loads(pending.read_text(encoding="utf-8"))
    assert local["revision_id"] != network["revision_id"]
    assert local["sha256"] != network["sha256"]
    assert local["split_group_sha256"] == network["split_group_sha256"]
    # Synthetic fixture approvals only: two versions cannot provide independent partitions.
    reviewed = [{**row, "visibility": "public", "privacy_reviewed": True, "training_allowed": True}
                for row in (local, network)]
    with pytest.raises(ValueError, match="Group split requires nonempty"):
        split_records(reviewed)


@pytest.mark.parametrize("change", [{"language": "es.evil.test"}, {"revision": True}, {"revision": "latest"}])
def test_invalid_spec_never_requests(tmp_path, change):
    def forbidden(request):
        pytest.fail("Invalid spec issued a request")
    with pytest.raises(ValueError):
        acquire([{**spec(), **change}], tmp_path / "pending", transport=httpx.MockTransport(forbidden))


@pytest.mark.parametrize("change", ["revision", "hidden", "namespace", "error", "oversized", "redirect"])
def test_invalid_response_does_not_publish(tmp_path, change):
    body = response()
    page = body["query"]["pages"][0]
    if change == "revision":
        page["revisions"][0]["revid"] = 456
    elif change == "hidden":
        del page["revisions"][0]["slots"]["main"]["content"]
    elif change == "namespace":
        page["ns"] = 2
    elif change == "error":
        body = {"error": {"code": "maxlag"}}
    def respond(request):
        if change == "oversized":
            return httpx.Response(200, content=b"x" * 500001)
        if change == "redirect":
            return httpx.Response(302, headers={"location": "https://example.invalid"})
        return httpx.Response(200, json=body)
    output = tmp_path / "pending"
    with pytest.raises((ValueError, httpx.HTTPStatusError)):
        acquire([spec()], output, transport=httpx.MockTransport(respond))
    assert not output.exists()
