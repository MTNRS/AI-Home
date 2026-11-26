import pytest
from test_nodes_and_jobs import client, register, login, create_node, user_headers
from ai_home.network_probe import measure_transfer


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
