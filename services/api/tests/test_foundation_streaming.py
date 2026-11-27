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


def test_real_http_disconnect_stops_generation_and_next_request_succeeds(tmp_path, monkeypatch):
    import socket
    import time
    import httpx
    import pytest
    import torch
    import uvicorn
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from ai_home.api import create_app
    from ai_home.config import Settings
    from ai_home.database import Base
    from ai_home.chat_stream import stream_server_chat, GenerationCancelled
    from ai_home.foundation.training import train, TrainingConfig
    import ai_home.foundation.inference as inference
    from test_foundation_model import config, records

    torch.set_num_threads(1)
    checkpoint = tmp_path / "real.pt"
    train(records(), checkpoint, config(), TrainingConfig(steps=1, warmup=0), "cpu")
    engine = create_engine("sqlite+pysqlite:///" + str(tmp_path / "stream.db"), connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    app = create_app(Settings(environment="test", foundation_checkpoint=checkpoint,
                              foundation_device="cpu", foundation_max_new_tokens=8), engine=engine)
    stopped = threading.Event()
    original = inference.stream_text
    def observed(*args, **kwargs):
        try:
            for piece in original(*args, **kwargs):
                yield piece
                # Hold after a real generated chunk to exercise in-flight disconnect,
                # not a stream that already completed before the socket closes.
                if kwargs.get("cancel") is not None:
                    kwargs["cancel"].wait(5)
        finally:
            if kwargs.get("cancel") is not None and kwargs["cancel"].is_set():
                stopped.set()
    monkeypatch.setattr(inference, "stream_text", observed)
    active_streams = []
    original_run = InferenceStream.run
    def tracked_run(stream):
        active_streams.append(stream)
        return original_run(stream)
    monkeypatch.setattr(InferenceStream, "run", tracked_run)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False, lifespan="off", ws="none"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=base, timeout=10) as client:
            credentials = {"email": "stream-test@example.com", "password": "strong-password-123"}
            assert client.post("/api/v1/auth/register", json=credentials).status_code == 201
            token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
            cancel, chunks = threading.Event(), []
            def receive(piece):
                chunks.append(piece)
                cancel.set()
            with pytest.raises(GenerationCancelled):
                stream_server_chat(base, token, [{"role": "user", "content": "Hi"}],
                                   cancel=cancel, on_delta=receive)
            assert chunks and stopped.wait(5)
            # The generator's finally runs before credit cleanup and lock release.
            # Wait for actual worker completion, not merely generator cancellation.
            assert active_streams and active_streams[-1].finished.wait(5)
            monkeypatch.setattr(inference, "stream_text", original)
            response = client.post("/api/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": f"Bearer {token}"})
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"].strip()
            from uuid import uuid4
            from sqlalchemy import select
            from ai_home.models import User
            from ai_home.credits import CreditLedger
            from ai_home.job_credits import user_account_id
            from ai_home.inference_credits import InferenceFunding
            with app.state.session_factory() as session:
                owner = user_account_id(session.scalar(select(User.user_id)))
            ledger = CreditLedger(engine)
            ledger.create_account("stream-test-issuance", "test", allow_negative=True)
            ledger.create_account(owner, "user")
            ledger.post(idempotency_key="stream-fund", operation="test", postings={"stream-test-issuance": -10, owner: 10})
            app.state.settings.foundation_inference_price = 7
            monkeypatch.setattr(inference, "stream_text", observed)
            stopped.clear()
            with client.stream("POST", "/api/v1/chat/stream", headers={"Authorization": f"Bearer {token}"},
                               json={"messages": [{"role": "user", "content": "Hi"}],
                                     "credit_request_id": str(uuid4()), "credit_limit": 7}) as paid_stream:
                assert paid_stream.status_code == 200
                for line in paid_stream.iter_lines():
                    if line.startswith("data: ") and json.loads(line[6:]).get("type") == "delta":
                        assert ledger.balance(owner) == 3
                        break
                else:
                    pytest.fail("No real paid-stream delta")
            assert stopped.wait(5)
            deadline = time.monotonic() + 5
            while ledger.balance(owner) != 10 and time.monotonic() < deadline:
                time.sleep(.01)
            assert ledger.balance(owner) == 10
            with app.state.session_factory() as session:
                assert session.scalar(select(InferenceFunding.status)) == "refunded"

    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        engine.dispose()
    assert not thread.is_alive()
