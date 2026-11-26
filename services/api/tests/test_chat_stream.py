import json
import threading
from uuid import uuid4

import httpx
import pytest

from ai_home.chat_stream import stream_server_chat, GenerationCancelled


def transport(events):
    def handle(request):
        assert request.headers["Authorization"] == "Bearer token"
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="".join("data: " + json.dumps(e) + "\n\n" for e in events))
    return httpx.MockTransport(handle)


def test_client_stream_protocol_and_version():
    version = str(uuid4())
    chunks = []
    result = stream_server_chat("https://example.org", "token", [{"role": "user", "content": "Hi"}],
        cancel=threading.Event(), on_delta=chunks.append, model_version=version,
        transport=transport([{"type": "metadata", "model_version": version, "references": []},
                             {"type": "delta", "content": "Hola"},
                             {"type": "done", "characters": 4, "chunks": 1}]))
    assert chunks == ["Hola"]
    assert result == {"answer": "Hola", "model_version": version, "references": []}


@pytest.mark.parametrize("events", [
    [{"type": "delta", "content": "before metadata"}],
    [{"type": "metadata"}, {"type": "delta", "content": "unfinished"}],
    [{"type": "metadata"}, {"type": "metadata"}],
    [{"type": "error", "code": "inference_failed"}],
    [{"type": "metadata"}, {"type": "done", "characters": 0, "chunks": 0}],
])
def test_incomplete_or_invalid_stream_is_never_success(events):
    with pytest.raises((ValueError, RuntimeError)):
        stream_server_chat("https://example.org", "token", [{"role": "user", "content": "Hi"}],
                           cancel=threading.Event(), on_delta=lambda _: None, transport=transport(events))


def test_cancel_while_waiting_for_response_headers():
    cancel = threading.Event()
    async def waiting(request):
        import asyncio
        cancel.set()
        await asyncio.Event().wait()
    with pytest.raises(GenerationCancelled):
        stream_server_chat("https://example.org", "token", [{"role": "user", "content": "Hi"}],
                           cancel=cancel, on_delta=lambda _: None, transport=httpx.MockTransport(waiting))
