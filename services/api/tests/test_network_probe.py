import pytest
import hashlib
import httpx
from test_nodes_and_jobs import client, register, login, create_node, user_headers
from ai_home.network_probe import measure_transfer


def test_worker_periodic_transfer_probe_and_failure_backoff(monkeypatch):
    from ai_home.node_worker import NodeAgent
    now, requests, failing = [0.], [], [False]
    monkeypatch.setattr("ai_home.node_worker.time.monotonic", lambda: now[0])

    def respond(request):
        requests.append(request.url.path)
        if request.url.path.endswith("heartbeat"):
            return httpx.Response(200, json={})
        if request.url.path.endswith("claim"):
            return httpx.Response(204)
        if failing[0]:
            return httpx.Response(429, text="private server detail")
        if request.url.path.endswith("transfer-measurement"):
            return httpx.Response(200, json={"received_epoch": 1})
        if request.method == "POST":
            return httpx.Response(200, json={"bytes": len(request.content),
                "sha256": hashlib.sha256(request.content).hexdigest()})
        size = int(request.url.params["size"])
        return httpx.Response(200, content=bytes(range(256)) * (size // 256))

    agent = NodeAgent(server_url="http://localhost", node_token="secret",
                      transport=httpx.MockTransport(respond), transfer_probe_interval=60)
    try:
        assert not agent.run_once()
        assert len(requests) == 8 and agent.last_transfer_measurement is not None
        now[0] = 59
        assert not agent.run_once() and len(requests) == 10
        now[0], failing[0] = 60, True
        assert not agent.run_once() and len(requests) == 13
        assert agent.last_transfer_measurement is None
        assert agent.last_transfer_probe_error == "HTTPStatusError"
        now[0] = 119
        assert not agent.run_once() and len(requests) == 15
        now[0], failing[0] = 120, False
        assert not agent.run_once() and len(requests) == 23
        assert agent.last_transfer_probe_error is None
    finally:
        agent.close()
    requests.clear()
    default = NodeAgent(server_url="http://localhost", node_token="secret",
                        transport=httpx.MockTransport(respond))
    try:
        assert not default.run_once() and len(requests) == 2
    finally:
        default.close()
    for interval in (True, 59, 121, 60.5):
        with pytest.raises(ValueError, match="Transfer probe interval"):
            NodeAgent(server_url="http://localhost", node_token="secret", transfer_probe_interval=interval)


def test_authenticated_transfer_and_limits(client):
    assert client.get("/api/v1/node/transfer-probe").status_code == 401
    assert client.post("/api/v1/node/transfer-probe", content=b"x").status_code == 401
    register(client, "probe@example.com")
    token = login(client, "probe@example.com")
    node = create_node(client, token)
    client.headers.update(user_headers(node["node_token"]))
    report = measure_transfer(client, size=65536)
    assert report["payload_bytes"] == 65536
    assert len(report["empty_round_trip_seconds"]) == 3
    assert all(x > 0 for x in report["empty_round_trip_seconds"])
    assert report["upload_bytes_per_second"] > 0 and report["download_bytes_per_second"] > 0
    assert client.get("/api/v1/node/transfer-probe", params={"size": 1048577}).status_code == 422
    assert client.post("/api/v1/node/transfer-probe", content=b"x" * 1048577).status_code == 413
    assert client.post("/api/v1/node/transfer-probe", content=b"x", headers={"content-encoding": "gzip"}).status_code == 415


def test_probe_requires_explicit_bounded_size():
    for size in (0, -1, True, 1048577):
        with pytest.raises(ValueError):
            measure_transfer(None, size=size)


def test_probe_rate_limit_returns_retry_after(client):
    register(client, "probe-limit@example.com")
    node = create_node(client, login(client, "probe-limit@example.com"))
    client.headers.update(user_headers(node["node_token"]))
    for _ in range(10):
        assert client.get("/api/v1/node/transfer-probe", params={"size": 0}).status_code == 200
    response = client.get("/api/v1/node/transfer-probe", params={"size": 0})
    assert response.status_code == 429 and response.headers["retry-after"] == "60"


def test_probe_budget_concurrent_admission_and_expiry():
    from concurrent.futures import ThreadPoolExecutor
    from ai_home.transfer_limits import TransferProbeLimiter
    now = [0.]
    limiter = TransferProbeLimiter(clock=lambda: now[0])
    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(lambda _: limiter.admit("node", 1048576), range(40)))
    assert sum(accepted) == 10
    for index in range(54):
        assert limiter.admit(str(index), 1048576)
    assert not limiter.admit("other", 1)
    now[0] = 60
    assert limiter.admit("node", 1048576) and len(limiter.events) == 1


def test_transfer_measurement_validated_stored_for_authenticated_node(client):
    from sqlalchemy import select
    from ai_home.models import ComputeNode
    register(client, "transfer-store@example.com")
    token = login(client, "transfer-store@example.com")
    first = create_node(client, token, "first")
    second = create_node(client, token, "second")
    sample = {"format": "ai-home-http-transfer-probe-v1", "payload_bytes": 65536,
              "empty_round_trip_seconds": [.01, .02, .03], "download_seconds": .1, "upload_seconds": .2}
    headers = user_headers(first["node_token"])
    response = client.post("/api/v1/node/transfer-measurement", json=sample, headers=headers)
    assert response.status_code == 200
    stored = response.json()
    assert stored["expires_epoch"] - stored["received_epoch"] == 120
    assert stored["download_bytes_per_second"] == 655360
    with client.app.state.session_factory() as session:
        nodes = {n.node_id: n for n in session.scalars(select(ComputeNode))}
        assert nodes[first["node_id"]].hardware["transfer_measurement"] == stored
        assert "transfer_measurement" not in nodes[second["node_id"]].hardware
    for patch in ({"received_epoch": 1}, {"payload_bytes": True}, {"upload_seconds": -1},
                  {"empty_round_trip_seconds": [.1]}):
        assert client.post("/api/v1/node/transfer-measurement", json={**sample, **patch}, headers=headers).status_code == 422
