import queue
import threading

import pytest
pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
from test_foundation_vision_checkpoint import saved
from ai_home.visual_desktop import run_visual, enqueue_chunk


def test_real_visual_stream_matches_final_and_cancel(saved, tmp_path):
    _, _, checkpoint, _, _ = saved
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(image)
    chunks = []
    text = run_visual(checkpoint, image, "Q", "cpu", threading.Event(), chunks.append)
    assert text == ("".join(chunks) or "El modelo no produjo texto.")
    cancel = threading.Event()
    cancel.set()
    assert run_visual("missing.pt", "missing.png", "Q", "cpu", cancel) == "Cancelado."


def test_full_stream_queue_stops_on_cancel():
    chunks = queue.Queue(maxsize=1)
    chunks.put("first")
    cancel = threading.Event()
    worker = threading.Thread(target=enqueue_chunk, args=(chunks, "second", cancel))
    worker.start()
    cancel.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert chunks.get_nowait() == "first" and chunks.empty()
