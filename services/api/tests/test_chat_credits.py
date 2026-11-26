import json
import threading
from uuid import uuid4

import httpx
import pytest

from ai_home.chat_credits import credit_authorization, validate_credit_receipt
from ai_home.desktop import server_chat_result, server_inference_price
from ai_home.chat_stream import stream_server_chat


@pytest.mark.parametrize("value", [True, -1, 1000001, "7", 0.5])
def test_spending_limit_is_strict(value):
    with pytest.raises(ValueError):
        credit_authorization(value)


def test_paid_client_requests_and_receipts():
    key = str(uuid4())
    auth = credit_authorization(7, key)
    assert credit_authorization() == {}
    assert credit_authorization(7)["credit_request_id"] != credit_authorization(7)["credit_request_id"]
    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"credits_per_completion": 7})
        body = json.loads(request.content)
        assert body["credit_limit"] == 7 and body["credit_request_id"] == key
        receipt = {"charged_credits": 7, "credit_request_id": key}
        if request.url.path.endswith("/stream"):
            events = [{"type":"metadata"}, {"type":"delta", "content":"Hola"},
                      {"type":"done", "characters":4, "chunks":1, **receipt}]
            return httpx.Response(200, headers={"content-type":"text/event-stream"},
                                  text="".join("data: " + json.dumps(row) + "\n\n" for row in events))
        return httpx.Response(200, json={"choices":[{"message":{"content":"Hola"}}], **receipt})
    transport = httpx.MockTransport(handle)
    args = ("https://example.org", "token", [{"role":"user", "content":"Hi"}])
    result = server_chat_result(*args, credit_limit=7, credit_request_id=key, transport=transport)
    assert result.charged_credits == 7
    streamed = stream_server_chat(*args, credit_limit=7, credit_request_id=key, transport=transport,
                                  cancel=threading.Event(), on_delta=lambda _: None)
    assert streamed["charged_credits"] == 7
    assert server_inference_price(args[0], args[1], transport=transport) == 7
    for receipt in ({"charged_credits":8}, {"charged_credits":True},
                    {"charged_credits":7, "credit_request_id":str(uuid4())}):
        with pytest.raises(ValueError):
            validate_credit_receipt(receipt, auth)
