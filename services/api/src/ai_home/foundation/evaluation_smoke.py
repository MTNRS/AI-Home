"""Real loopback evaluation with two worker processes and restarted coordinator."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time

import httpx
import torch
import uvicorn
from alembic import command

from ai_home.api import create_app
from ai_home.config import Settings
from ai_home.credit_admin import grant_credits
from ai_home.migrate import migration_config
from ai_home.node_worker import NodeAgent, detect_hardware
from .continual import checkpoint_digest, compare_checkpoint
from .data import prepare_records
from .evaluation_coordinator import create_plan
from .model import FoundationConfig
from .training import TrainingConfig, train


def run(output, devices):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    python_command = [sys.executable, *(["-I"] if sys.flags.isolated else [])]
    records = [{"source": "authored evaluation smoke", "url": "local:evaluation-smoke",
        "license": "Apache-2.0", "date": "2026-11-26", "language": "es",
        "training_allowed": True, "privacy_reviewed": True, "visibility": "public",
        "text": f"Documento {i}: Evaluamos un modelo propio con datos autorizados. " * 3}
        for i in range(8)]
    validation = prepare_records(records[4:])
    config = FoundationConfig(vocab_size=280, width=16, layers=1, heads=2, kv_heads=1, hidden=32, context=16)
    training = TrainingConfig(steps=2, warmup=0)
    checkpoint = output / "model.pt"
    trained = train(records[:4], checkpoint, config, training, "cpu", validation_records=validation)
    identity = checkpoint_digest(checkpoint)
    reference = compare_checkpoint(checkpoint, validation, "causal", "cpu", 10_000_000)
    (output / "dataset.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "configuration.json").write_text(json.dumps({"model": asdict(config),
        "training": asdict(training), "worker_devices": devices}, indent=2), encoding="utf-8")
    settings = Settings(_env_file=None, environment="test",
        database_url=f"sqlite+pysqlite:///{(output / 'jobs.db').as_posix()}",
        supabase_url=None, supabase_publishable_key="", supabase_jwks_url=None,
        foundation_distributed_evaluation_enabled=True)
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
            raise RuntimeError("Evaluation server failed to start")
        def account(name):
            client = httpx.Client(base_url=url, timeout=60, trust_env=False)
            credentials = {"email": name + "@example.com", "password": secrets.token_urlsafe(32)}
            response = client.post("/api/v1/auth/register", json=credentials)
            response.raise_for_status()
            user = response.json()
            response = client.post("/api/v1/auth/login", json=credentials)
            response.raise_for_status()
            token = response.json()["access_token"]
            client.headers["Authorization"] = "Bearer " + token
            return client, user, token
        owner, user, token = account("evaluation-owner")
        clients = [owner]
        try:
            with app.state.session_factory() as session, session.begin():
                grant_credits(session, user_id=user["user_id"], amount=6,
                              idempotency_key="smoke-issuance", reason="isolated evaluation smoke")
            plan, report_path = output / "plan.json", output / "evaluation.json"
            create_plan(checkpoint, [validation[:2], validation[2:]], plan, server=url,
                        owner_user_id=user["user_id"], reward_credits=3,
                        allow_public_node_processing=True, weights_public_and_redistributable=True)
            def coordinate():
                result = subprocess.run(python_command + ["-m", "ai_home.foundation.evaluation_coordinator",
                    "advance", "--checkpoint", str(checkpoint), "--plan", str(plan),
                    "--server", url, "--output", str(report_path)],
                    env={**os.environ, "AI_HOME_COORDINATOR_TOKEN": token},
                    capture_output=True, text=True, timeout=90)
                if result.returncode:
                    raise RuntimeError("Evaluation coordinator process failed: " + result.stderr[-2000:])
                return json.loads(result.stdout)
            waiting = coordinate()
            assert waiting["state"] == "waiting" and len(waiting["jobs"]) == 2
            worker_reports, balances = [], []
            for index, device in enumerate(devices):
                client, _, _ = account(f"evaluation-worker-{index}")
                clients.append(client)
                response = client.post("/api/v1/nodes", json={"name": f"worker-{index}",
                    "hardware": detect_hardware().model_dump(mode="json")})
                response.raise_for_status()
                node_token = response.json()["node_token"]
                result = subprocess.run(python_command + ["-m", "ai_home.foundation.evaluation_smoke", "--worker"],
                    env={**os.environ, "AI_HOME_SMOKE_SERVER": url, "AI_HOME_SMOKE_NODE_TOKEN": node_token,
                         "AI_HOME_SMOKE_DEVICE": device}, capture_output=True, text=True, timeout=90)
                if result.returncode:
                    raise RuntimeError("Evaluation worker process failed: " + result.stderr[-2000:])
                worker_reports.append(json.loads(result.stdout))
                response = client.get("/api/v1/credits")
                response.raise_for_status()
                balances.append(response.json())
            evaluated = coordinate()
            assert coordinate() == evaluated
            assert evaluated["state"] == "evaluated" and evaluated["tokens"] == reference["tokens"]
            assert math.isclose(evaluated["mean_loss"], reference["loss"], rel_tol=1e-5, abs_tol=1e-6)
            assert all(b == {"available": 3, "reserved": 0, "earned": 3} for b in balances)
            owner_balance = owner.get("/api/v1/credits").json()
            assert owner_balance["available"] == owner_balance["reserved"] == 0
            jobs = owner.get("/api/v1/jobs").json()
            assert len(jobs) == 2 and all(j["status"] == "completed" for j in jobs)
            assert checkpoint_digest(checkpoint) == identity
            report = {"format": "ai-home-evaluation-smoke-v1", "transport": "real loopback HTTP",
                "workers": worker_reports, "worker_balances": balances, "owner_balance": owner_balance,
                "coordinator_processes": 3, "completed_jobs": len(jobs), "checkpoint_sha256": identity,
                "training_embedding_delta_l2": trained["embedding_delta_l2"], "reference": reference,
                "distributed_loss": evaluated["mean_loss"], "tokens": evaluated["tokens"],
                "scope": "One machine, two worker processes; no multi-host or general capability claim"}
            (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report
        finally:
            for client in clients:
                client.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        app.state.engine.dispose()
        if thread.is_alive():
            raise RuntimeError("Evaluation server did not stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-devices", nargs=2, choices=("cpu", "cuda"), default=["cpu", "cpu"])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        torch.set_num_threads(1)
        agent = NodeAgent(server_url=os.environ["AI_HOME_SMOKE_SERVER"], node_token=os.environ["AI_HOME_SMOKE_NODE_TOKEN"],
                          allow_foundation_evaluation=True, foundation_device=os.environ["AI_HOME_SMOKE_DEVICE"])
        try:
            agent.configure_capabilities()
            if not agent.run_once() or not agent.last_measurement_uploaded:
                raise RuntimeError("Worker did not complete and report evaluation")
            print(json.dumps(agent.last_run_metrics))
        finally:
            agent.close()
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required")
        print(json.dumps(run(args.output_dir, args.worker_devices), indent=2))


if __name__ == "__main__":
    main()
