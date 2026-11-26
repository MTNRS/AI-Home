"""Two real worker processes tokenizing public shards over loopback HTTP."""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time

import httpx
import uvicorn

from ai_home.api import create_app
from ai_home.config import Settings
from ai_home.database import Base
from ai_home.node_worker import NodeAgent, detect_hardware
from .tasks import execute_tokenize
from .tokenizer import ByteBPE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        config = json.load(sys.stdin)
        agent = NodeAgent(server_url=config["url"], node_token=config["token"],
                          allow_foundation_training=config.get("allow_foundation_training", False),
                          foundation_device=config.get("foundation_device", "cpu"))
        try:
            transfer = agent.measure_transfer(publish=True) if config.get("transfer_probe") else None
            completed = agent.run_once()
            print(json.dumps({"pid": os.getpid(), "completed": completed, "measurement": agent.last_run_metrics,
                              "measurement_uploaded": agent.last_measurement_uploaded, "transfer_measurement": transfer}))
        finally:
            agent.close()
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = Settings(_env_file=None, environment="test", database_url=f"sqlite+pysqlite:///{(output / 'jobs.db').as_posix()}",
                        supabase_url=None, supabase_publishable_key="", supabase_jwks_url=None)
    app = create_app(settings)
    Base.metadata.create_all(app.state.engine)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    workers, shards = [], []
    try:
        for _ in range(100):
            if server.started:
                break
            if not thread.is_alive():
                raise RuntimeError("Loopback server stopped")
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError("Loopback server failed to start")
        url = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=url, timeout=30) as client:
            credentials = {"email": "node-smoke@example.com", "password": secrets.token_urlsafe(32)}
            client.post("/api/v1/auth/register", json=credentials).raise_for_status()
            response = client.post("/api/v1/auth/login", json=credentials)
            response.raise_for_status()
            client.headers["Authorization"] = "Bearer " + response.json()["access_token"]
            records = [{"source": "AI@Home authored node fixture", "url": "local:node_smoke",
                        "license": "Apache-2.0", "date": "2026-11-26", "language": "es",
                        "training_allowed": True, "privacy_reviewed": True, "visibility": "public",
                        "text": f"Documento {i}: un nodo convierte texto autorizado en tokens."}
                       for i in range(4)]
            tokenizer = ByteBPE.train([r["text"] for r in records], vocab_size=300)
            tokenizer.save(output / "tokenizer.json")
            hardware = detect_hardware().model_dump(mode="json")
            for i in range(2):
                payload = {"records": records[i * 2:i * 2 + 2], "tokenizer": tokenizer.to_dict()}
                shards.append(payload)
                response = client.post("/api/v1/jobs/foundation/tokenize", json={"payload": payload,
                    "allow_public_node_processing": True, "idempotency_key": f"shard-{i}"})
                response.raise_for_status()
            for i in range(2):
                response = client.post("/api/v1/nodes", json={"name": f"worker-{i}", "hardware": hardware,
                                      "supported_job_types": ["foundation.tokenize.v1"]})
                response.raise_for_status()
                registration = response.json()
                process = subprocess.run([sys.executable, "-m", "ai_home.foundation.node_smoke", "--worker"],
                    input=json.dumps({"url": url, "token": registration["node_token"]}), text=True,
                    capture_output=True, check=True, timeout=60,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                worker = json.loads(process.stdout)
                if not worker["completed"]:
                    raise RuntimeError("Worker did not complete a shard")
                workers.append({"node_id": registration["node_id"], **worker})
            response = client.get("/api/v1/jobs")
            response.raise_for_status()
            jobs = response.json()
            expected = {execute_tokenize(p)["artifact_sha256"] for p in shards}
            if len(jobs) != 2 or any(j["status"] != "completed" for j in jobs):
                raise RuntimeError("Not all shards completed")
            if {j["result"]["artifact_sha256"] for j in jobs} != expected:
                raise RuntimeError("Token artifacts differ from verified reference")
            for i, job in enumerate(jobs):
                (output / f"shard-{i}.json").write_text(job["result"]["artifact_json"], encoding="utf-8")
            report = {"format": "ai-home-node-tokenization-smoke-v1", "transport": "loopback HTTP",
                "controller_pid": os.getpid(), "workers": workers, "job_type": "foundation.tokenize.v1",
                "completed_jobs": len(jobs), "tokenizer_sha256": tokenizer.fingerprint,
                "artifact_sha256": sorted(expected), "tokens": sum(j["result"]["token_count"] for j in jobs),
                "claim": "Two worker processes on one computer; data preparation, not distributed neural training"}
            if len({w["pid"] for w in workers}) != 2:
                raise RuntimeError("Workers must be distinct processes")
            (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        app.state.engine.dispose()


if __name__ == "__main__":
    main()
