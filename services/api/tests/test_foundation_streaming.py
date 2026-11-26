import asyncio
import json
import threading

from ai_home.foundation.streaming import InferenceStream


def test_disconnect_cancels_worker_and_releases_lock_after_work_stops():
    lock = threading.Lock()
    stopped = threading.Event()
    def work(cancel, emit):
        emit({"type": "metadata"})
        cancel.wait(2)
        assert cancel.is_set()
        assert lock.locked()
        stopped.set()
    stream = InferenceStream(work, lock)
    async def consume():
        events = stream.events()
        assert "metadata" in await anext(events)
        await events.aclose()
        assert await asyncio.to_thread(stream.finished.wait, 3)
    asyncio.run(consume())
    assert stopped.is_set() and not lock.locked()
    assert stream.queue.maxsize == 16


def test_stream_errors_are_redacted_and_busy_never_runs_work():
    async def collect(stream):
        return [json.loads(event.removeprefix("data: ")) async for event in stream.events()]
    def failure(cancel, emit):
        raise ValueError("private prompt and C:/secret/checkpoint.pt")
    lock = threading.Lock()
    assert asyncio.run(collect(InferenceStream(failure, lock))) == [{"type": "error", "code": "inference_failed"}]
    assert not lock.locked()
    lock.acquire()
    try:
        assert asyncio.run(collect(InferenceStream(failure, lock))) == [{"type": "error", "code": "busy"}]
        assert lock.locked()
    finally:
        lock.release()
