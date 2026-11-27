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
from .data import canonical_hash, prepare_records
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
    holdout_path = output / "holdout.json"
    holdout_version = canonical_hash(validation)
    holdout_path.write_text(json.dumps({"format_version": 1, "dataset_version": holdout_version,
                                      "records": validation}, indent=2), encoding="utf-8")
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
        observer = None
        try:
            with app.state.session_factory() as session, session.begin():
                grant_credits(session, user_id=user["user_id"], amount=6,
                              idempotency_key="smoke-issuance", reason="isolated evaluation smoke")
            plan, report_path = output / "plan.json", output / "evaluation.json"
            planned = subprocess.run(python_command + ["-m", "ai_home.foundation.evaluation_coordinator",
                "plan", "--checkpoint", str(checkpoint), "--dataset", str(holdout_path),
                "--shard-count", "2", "--output", str(plan), "--server", url,
                "--owner", user["user_id"], "--reward-credits", "3",
                "--allow-public-node-processing", "--weights-public-and-redistributable"],
                capture_output=True, text=True, timeout=90)
            if planned.returncode:
                raise RuntimeError("Evaluation planning process failed: " + planned.stderr[-2000:])
            planning = json.loads(planned.stdout)
            stored_plan = json.loads(plan.read_text(encoding="utf-8"))
            assert planning == {"state": "planned", "plan_sha256": stored_plan["sha256"], "shards": 2}
            planned_records = prepare_records([record for payload in stored_plan["payloads"]
                                               for record in payload["shard"]["records"]])
            assert {record["sha256"] for record in planned_records} == {record["sha256"] for record in validation}
            assert owner.get("/api/v1/jobs").json() == []
            assert owner.get("/api/v1/credits").json()["reserved"] == 0
            coordinator_command = python_command + ["-m", "ai_home.foundation.evaluation_coordinator",
                "advance", "--checkpoint", str(checkpoint), "--plan", str(plan),
                "--server", url, "--output", str(report_path)]
            coordinator_environment = {**os.environ, "AI_HOME_COORDINATOR_TOKEN": token}
            def coordinate():
                result = subprocess.run(coordinator_command, env=coordinator_environment,
                                        capture_output=True, text=True, timeout=90)
                if result.returncode:
                    raise RuntimeError("Evaluation coordinator process failed: " + result.stderr[-2000:])
                return json.loads(result.stdout)
            waiting = coordinate()
            assert waiting["state"] == "waiting" and len(waiting["jobs"]) == 2
            observer = subprocess.Popen(coordinator_command + ["--wait-seconds", "60", "--poll-seconds", "1"],
                env=coordinator_environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
            observed_stdout, observed_stderr = observer.communicate(timeout=90)
            if observer.returncode:
                raise RuntimeError("Evaluation observer failed: " + observed_stderr[-2000:])
            observed = json.loads(observed_stdout)
            assert observed["state"] == "evaluated"
            observation = {key: observed.pop(key) for key in ("attempts", "transport_errors")}
            evaluated = observed
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
                "coordinator_processes": 4, "planning": planning, "holdout_dataset_sha256": holdout_version,
                "observation": observation, "completed_jobs": len(jobs), "checkpoint_sha256": identity,
                "training_embedding_delta_l2": trained["embedding_delta_l2"], "reference": reference,
                "distributed_loss": evaluated["mean_loss"], "tokens": evaluated["tokens"],
                "scope": "One machine, two worker processes; no multi-host or general capability claim"}
            (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report
        finally:
            if observer is not None and observer.poll() is None:
                observer.terminate()
                observer.communicate(timeout=10)
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
