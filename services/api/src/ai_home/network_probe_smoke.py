"""Real loopback transfer probe, with isolated migrated DB and synthetic node."""
import argparse
import json
from pathlib import Path
import secrets
import socket
import threading
import time

from alembic import command
import httpx
import uvicorn
from sqlalchemy import select, func

from .api import create_app
from .config import Settings
from .credits import LedgerTransaction
from .migrate import migration_config
from .models import ComputeJob
from .node_worker import NodeAgent


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = Settings(_env_file=None, environment="test",
        database_url=f"sqlite+pysqlite:///{(output / 'probe.db').as_posix()}",
        supabase_url=None, supabase_publishable_key="", supabase_jwks_url=None)
    command.upgrade(migration_config(settings), "head")
    app = create_app(settings)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.05)
        if not server.started:
            raise RuntimeError("Probe server failed to start")
        with httpx.Client(base_url=url, timeout=10) as client:
            credentials = {"email": "transfer-smoke@example.com", "password": secrets.token_urlsafe(32)}
            client.post("/api/v1/auth/register", json=credentials).raise_for_status()
            response = client.post("/api/v1/auth/login", json=credentials)
            response.raise_for_status()
            client.headers["Authorization"] = "Bearer " + response.json()["access_token"]
            response = client.post("/api/v1/nodes", json={"name": "transfer-smoke",
                "hardware": {"platform": "synthetic", "cpu_model": "fixture", "logical_cores": 1,
                             "ram_total_bytes": 1073741824, "gpus": []},
                "supported_job_types": ["math.sum.v1"]})
            response.raise_for_status()
            node = response.json()
        def counts():
            with app.state.engine.connect() as connection:
                return {"jobs": connection.scalar(select(func.count()).select_from(ComputeJob)),
                        "credit_transactions": connection.scalar(select(func.count()).select_from(LedgerTransaction))}
        before = counts()
        agent = NodeAgent(server_url=url, node_token=node["node_token"])
        try:
            samples = [agent.measure_transfer(size) for size in (65536, 1048576)]
        finally:
            agent.close()
        after = counts()
        if before != after:
            raise RuntimeError("Transfer probe altered jobs or credit transactions")
        report = {"format": "ai-home-transfer-smoke-v1", "transport": "real loopback HTTP",
            "samples": samples, "before": before, "after": after,
            "claim": "One machine, synthetic account; no physical-network bandwidth or multi-host evidence"}
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        app.state.engine.dispose()
        if thread.is_alive():
            raise RuntimeError("Probe server did not stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
