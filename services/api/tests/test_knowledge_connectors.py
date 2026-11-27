import json
import threading

import httpx
import pytest

from ai_home.knowledge_connectors import acquire
from ai_home.knowledge_ingest import create_review_template, approve_records
from ai_home.foundation.data import load_records


@pytest.mark.parametrize("connector", ["github", "wikipedia"])
def test_cancellation_closes_download_without_publishing(tmp_path, connector):
    from ai_home.knowledge_connectors import AcquisitionCancelled
    cancel = threading.Event()
    closed = []
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"first block"
            cancel.set()
            yield b"second block"
        def close(self):
            closed.append(True)
    source = spec() if connector == "github" else dict(connector="wikipedia-revision-v1",
        language="es", revision=123, license="declared fixture license")
    output = tmp_path / "pending.jsonl"
    with pytest.raises(AcquisitionCancelled):
        acquire([source], output, cancel=cancel,
                transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Stream())))
    assert closed and not output.exists()
    assert not list(tmp_path.glob("*.artifact.tmp"))


def test_cancelled_batch_does_not_request_or_publish_after_flush(tmp_path, monkeypatch):
    from ai_home import knowledge_connectors as connectors
    cancel = threading.Event()
    cancel.set()
    def forbidden(request):
        pytest.fail("Cancelled acquisition made a request")
    output = tmp_path / "pending.jsonl"
    with pytest.raises(connectors.AcquisitionCancelled):
        acquire([spec()], output, cancel=cancel, transport=httpx.MockTransport(forbidden))
    cancel.clear()
    original = connectors.publish_bytes
    def delayed(path, raw, *, before_publish):
        def after_flush():
            cancel.set()
            before_publish()
        return original(path, raw, before_publish=after_flush)
    monkeypatch.setattr(connectors, "publish_bytes", delayed)
    with pytest.raises(connectors.AcquisitionCancelled):
        acquire([spec()], output, cancel=cancel,
                transport=httpx.MockTransport(lambda r: httpx.Response(200, text="Public fixture text")))
    assert not output.exists() and not list(tmp_path.glob("*.artifact.tmp"))


def spec(path="README.md"):
    return dict(connector="github-text-v1", repository="example/project", commit="a" * 40,
                path=path, license="Apache-2.0", language="es")


def test_github_versions_share_file_group_but_other_paths_do_not(tmp_path):
    pending = tmp_path / 'versions.jsonl'
    acquire([spec(), {**spec(), 'commit': 'b' * 40, 'repository': 'Example/Project'}, spec('other.md')],
            pending, transport=httpx.MockTransport(lambda r: httpx.Response(200, text=str(r.url))))
    rows = [json.loads(line) for line in pending.read_text().splitlines()]
    assert rows[0]['split_group_sha256'] == rows[1]['split_group_sha256']
    assert rows[0]['split_group_sha256'] != rows[2]['split_group_sha256']


def test_public_connector_requires_review_before_use(tmp_path):
    seen = []
    def respond(request):
        seen.append(request)
        return httpx.Response(200, content=("Contenido autorizado " + request.url.path + "\r\n").encode())
    pending = tmp_path / "pending.jsonl"
    result = acquire([spec(), spec("docs/other.md")], pending, transport=httpx.MockTransport(respond))
    assert result["documents"] == 2
    assert all(r.url.host == "raw.githubusercontent.com" and "authorization" not in r.headers for r in seen)
    rows = [json.loads(line) for line in pending.read_text(encoding="utf-8").splitlines()]
    assert all(not r["training_allowed"] and not r["retrieval_allowed"] and not r["privacy_reviewed"] for r in rows)
    assert rows[0]["source_sha256"] != rows[0]["sha256"]
    review = tmp_path / "review.json"
    create_review_template(pending, review)
    approved = tmp_path / "dataset.json"
    with pytest.raises(ValueError):
        approve_records(pending, review, approved)
    data = json.loads(review.read_text(encoding="utf-8"))
    data.update(reviewer="fixture reviewer", date="2026-11-26")
    for row in data["records"]:
        row.update(privacy_reviewed=True, license_reviewed=True, retrieval_allowed=True)
    review.write_text(json.dumps(data), encoding="utf-8")
    approve_records(pending, review, approved)
    retained = load_records(approved, purpose="retrieval")[0]
    assert len(retained) == 2
    assert {r['split_group_sha256'] for r in retained} == {r['split_group_sha256'] for r in rows}
    with pytest.raises(ValueError, match="Training authorization"):
        load_records(approved)


@pytest.mark.parametrize("change", [{"commit": "main"}, {"path": "../secret.txt"},
                                    {"repository": "https://example.com"}, {"path": "file.exe"},
                                    {"connector": "arbitrary.module"}])
def test_connector_rejects_unsafe_or_mutable_specs(tmp_path, change):
    def forbidden(request):
        pytest.fail("Invalid source must not issue a request")
    with pytest.raises(ValueError):
        acquire([{**spec(), **change}], tmp_path / "output", transport=httpx.MockTransport(forbidden))


@pytest.mark.parametrize("response", [httpx.Response(302, headers={"location": "http://localhost/private"}),
                                      httpx.Response(200, content=b"x" * 500001),
                                      httpx.Response(200, content=b"\x00bad"), httpx.Response(403)])
def test_download_failures_publish_no_pending_batch(tmp_path, response):
    path = tmp_path / "output"
    with pytest.raises((ValueError, httpx.HTTPError)):
        acquire([spec()], path, transport=httpx.MockTransport(lambda _: response))
    assert not path.exists()
