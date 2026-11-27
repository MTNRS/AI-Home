"""Bounded SSE transport for cancellable local Foundation generation."""
import asyncio
import json
import queue
import threading


class InferenceStream:
    def __init__(self, work, lock):
        self.work, self.lock = work, lock
        self.cancel = threading.Event()
        self.finished = threading.Event()
        self.queue = queue.Queue(maxsize=16)

    def emit(self, event):
        while not self.cancel.is_set():
            try:
                self.queue.put(event, timeout=0.1)
                return
            except queue.Full:
                continue

    def run(self):
        acquired = False
        try:
            acquired = self.lock.acquire(blocking=False)
            if not acquired:
                self.emit({"type": "error", "code": "busy"})
                return
            self.work(self.cancel, self.emit)
        except Exception:
            # Never expose paths, checkpoint bytes or private prompt contents.
            self.emit({"type": "error", "code": "inference_failed"})
        finally:
            if acquired:
                self.lock.release()
            self.finished.set()
            self.emit(None)

    async def events(self):
        worker = threading.Thread(target=self.run, daemon=True)
        worker.start()
        try:
            while True:
                try:
                    event = await asyncio.to_thread(self.queue.get, True, 0.1)
                except queue.Empty:
                    continue
                if event is None:
                    return
                yield "data: " + json.dumps(event, ensure_ascii=True) + "\n\n"
        finally:
            # The worker releases the inference lock only after generation stops.
            self.cancel.set()
