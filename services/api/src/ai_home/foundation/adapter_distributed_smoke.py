"""Real loopback adapter work with two separately opted-in worker processes."""
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
import torch
import uvicorn
from alembic import command
from ai_home.api import create_app
from ai_home.config import Settings
from ai_home.credit_admin import grant_credits
from ai_home.migrate import migration_config
from ai_home.node_worker import NodeAgent, detect_hardware
from .adapters import attach_adapters, save_adapter, export_merged_adapter
from .adapter_tasks import aggregate_adapter_gradients
from .checkpoint import load_checkpoint
from .distributed import encode_tensors
from .model import FoundationConfig
from .training import train, TrainingConfig


def run(output, devices, *, context=16, parallel=False):
    if type(context) is not int or not 8 <= context <= 128:
        raise ValueError("Context must be an integer between 8 and 128")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    records = [{"source": "authored adapter network fixture", "url": "local:adapter-smoke",
        "license": "Apache-2.0", "date": "2026-11-26", "language": "es", "visibility": "public",
        "training_allowed": True, "privacy_reviewed": True,
        "text": f"Documento {i}: Un modelo propio aprende con datos autorizados."} for i in range(6)]
    (output / "dataset.json").write_text(json.dumps({"format_version": 1, "records": records}), encoding="utf-8")
    parent_path = output / "parent.pt"
    train(records[:4], parent_path, FoundationConfig(vocab_size=280, width=16, layers=1,
          heads=2, kv_heads=1, hidden=32, context=context), TrainingConfig(steps=2, warmup=0),
          "cpu", validation_records=records[4:])
    base, tokenizer, _ = load_checkpoint(parent_path)
    model = attach_adapters(base, rank=2, alpha=2.)
    factors = {k:v for k,v in model.named_parameters() if v.requires_grad}
    before = {k:v.clone() for k,v in factors.items()}
    payloads = [dict(base={"shard": {"records": records[i:i+2], "tokenizer": tokenizer.to_dict()},
        "config": base.config.to_dict(), "parameters": encode_tensors(dict(base.named_parameters())),
        "torch_version": str(torch.__version__), "weights_public_and_redistributable": True},
        rank=2, alpha=2., factors=encode_tensors(factors), factors_public_and_redistributable=True) for i in (0,2)]
    settings = Settings(_env_file=None, environment="test",
        database_url=f"sqlite+pysqlite:///{(output / 'jobs.db').as_posix()}",
        supabase_url=None, supabase_publishable_key="", supabase_jwks_url=None,
        foundation_distributed_adapters_enabled=True)
    command.upgrade(migration_config(settings), "head")
    app = create_app(settings)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    clients = []
    processes = []
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.05)
        if not server.started:
            raise RuntimeError("Adapter server failed to start")
        def account(name):
            client = httpx.Client(base_url=url, timeout=60, trust_env=False)
            clients.append(client)
            credentials = {"email": name + "@example.com", "password": secrets.token_urlsafe(32)}
            response = client.post("/api/v1/auth/register", json=credentials)
            response.raise_for_status()
            user = response.json()
            response = client.post("/api/v1/auth/login", json=credentials)
            response.raise_for_status()
            client.headers["Authorization"] = "Bearer " + response.json()["access_token"]
            return client, user
        owner, user = account("adapter-owner")
        with app.state.session_factory() as session, session.begin():
            grant_credits(session, user_id=user["user_id"], amount=6,
                          idempotency_key="adapter-smoke-issuance", reason="isolated authored adapter experiment")
        ids = []
        for index, payload in enumerate(payloads):
            response = owner.post("/api/v1/jobs/foundation/adapter-gradient", json={"payload": payload,
                "idempotency_key": f"adapter-{index}", "allow_public_node_processing": True, "reward_credits": 3})
            response.raise_for_status()
            ids.append(response.json()["job_id"])
        workers, balances = [], []
        pending = []
        def collect(process, client):
            stdout, stderr = process.communicate(timeout=90)
            if process.returncode:
                raise RuntimeError("Adapter worker failed: " + stderr[-2000:])
            workers.append(json.loads(stdout))
            balances.append(client.get("/api/v1/credits").json())
        for index, device in enumerate(devices):
            client, _ = account(f"adapter-worker-{index}")
            response = client.post("/api/v1/nodes", json={"name": f"adapter-worker-{index}",
                "hardware": detect_hardware().model_dump(mode="json")})
            response.raise_for_status()
            process = subprocess.Popen([sys.executable, *(["-I"] if sys.flags.isolated else []),
                "-m", "ai_home.foundation.adapter_distributed_smoke", "--worker"],
                env={**os.environ, "AI_HOME_SMOKE_SERVER": url,
                     "AI_HOME_SMOKE_NODE_TOKEN": response.json()["node_token"], "AI_HOME_SMOKE_DEVICE": device},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(process)
            if parallel:
                pending.append((process, client))
            else:
                collect(process, client)
        for process, client in pending:
            collect(process, client)
        jobs = {job["job_id"]: job for job in owner.get("/api/v1/jobs").json()}
        assert all(jobs[key]["status"] == "completed" for key in ids)
        optimizer = torch.optim.AdamW(list(factors.values()), lr=.001)
        stats = aggregate_adapter_gradients(model, tokenizer, payloads, [jobs[key]["result"] for key in ids], optimizer)
        delta = sum((factors[k].detach() - value).square().sum().item() for k,value in before.items()) ** .5
        assert delta > 0 and all(b == {"available": 3, "reserved": 0, "earned": 3} for b in balances)
        owner_balance = owner.get("/api/v1/credits").json()
        assert owner_balance == {"available": 0, "reserved": 0, "earned": 0}
        save_adapter(model, parent_path, output / "adapter.pt", training_records=records[:4], validation_records=records[4:])
        export_merged_adapter(parent_path, output / "adapter.pt", output / "candidate.pt")
        report = {"format": "ai-home-adapter-distributed-smoke-v1", "transport": "real loopback HTTP",
            "context": context,
            "worker_launch": "concurrent" if parallel else "sequential",
            "workers": workers, "worker_balances": balances, "owner_balance": owner_balance,
            "completed_jobs": 2, "aggregation": stats, "factor_delta_l2": delta,
            "promoted": False, "scope": "One machine, two worker processes; no multihost or efficiency claim"}
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate()
        for client in clients:
            client.close()
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        app.state.engine.dispose()
        if thread.is_alive():
            raise RuntimeError("Adapter server did not stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--parallel", action="store_true", help="Launch both worker processes before waiting")
    parser.add_argument("--worker-devices", nargs=2, choices=("cpu", "cuda"), default=["cpu", "cpu"])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        torch.set_num_threads(1)
        agent = NodeAgent(server_url=os.environ["AI_HOME_SMOKE_SERVER"], node_token=os.environ["AI_HOME_SMOKE_NODE_TOKEN"],
            allow_foundation_adapters=True, foundation_device=os.environ["AI_HOME_SMOKE_DEVICE"])
        try:
            agent.configure_capabilities()
            if not agent.run_once():
                raise RuntimeError("Adapter worker did not complete a job")
            print(json.dumps({"pid": os.getpid(), "device": agent.foundation_device, "metrics": agent.last_run_metrics}))
        finally:
            agent.close()
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required")
        print(json.dumps(run(args.output_dir, args.worker_devices, context=args.context, parallel=args.parallel), indent=2))


if __name__ == "__main__":
    main()
