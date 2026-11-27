from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from ai_home.immutable_artifacts import publish_bytes


def test_competing_publishers_preserve_one_complete_version(tmp_path):
    target = tmp_path / "version.json"
    barrier = threading.Barrier(2)
    values = [b"a" * 100000, b"b" * 100000]
    def publish(value):
        try:
            publish_bytes(target, value, before_publish=lambda: barrier.wait(timeout=5))
            return True
        except FileExistsError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, values))
    assert sorted(outcomes) == [False, True]
    assert target.read_bytes() == values[outcomes.index(True)]
    assert list(tmp_path.iterdir()) == [target]


def test_flush_failure_and_prepublication_cancellation_leave_no_target(tmp_path, monkeypatch):
    import ai_home.immutable_artifacts as artifacts
    target = tmp_path / "version.json"
    def fail(*args):
        raise OSError("injected disk failure")
    with monkeypatch.context() as patch:
        patch.setattr(artifacts.os, "fsync", fail)
        with pytest.raises(OSError, match="disk failure"):
            publish_bytes(target, b"complete")
    assert not list(tmp_path.iterdir())
    with pytest.raises(OSError, match="disk failure"):
        publish_bytes(target, b"complete", before_publish=fail)
    assert not list(tmp_path.iterdir())
