"""Two real HTTP workers contribute gradients to a shared Foundation model."""
import argparse
from dataclasses import asdict
import hashlib
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

from ai_home.api import create_app
from ai_home.config import Settings
from ai_home.database import Base
from ai_home.credits import CreditLedger
from ai_home.job_credits import user_account_id
from ai_home.node_worker import detect_hardware
from .checkpoint import FORMAT, atomic_save, load_checkpoint
from .data import canonical_hash, prepare_records
from .distributed import aggregate_gradients, encode_tensors
from .inference import stream_text
from .model import FoundationConfig, FoundationModel
from .tokenizer import ByteBPE
from .training import _sequences, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--reward-credits", type=int, default=0)
    parser.add_argument("--worker-devices", nargs=2, choices=("cpu", "cuda"), default=["cpu", "cpu"])
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10:
        parser.error("Reference experiment permits 1-10 rounds")
    if not 0 <= args.reward_credits <= 100:
        parser.error("Smoke reward must be between 0 and 100 internal test credits")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    records = [{"source": "AI@Home authored distributed fixture", "url": "local:distributed_smoke",
        "license": "Apache-2.0", "date": "2026-11-26", "language": "es",
        "training_allowed": True, "privacy_reviewed": True, "visibility": "public",
        "text": f"Documento {i}: Un modelo propio aprende con datos autorizados."} for i in range(6)]
    train_records, validation = records[:4], prepare_records(records[4:])
    tokenizer = ByteBPE.train([r["text"] for r in train_records], vocab_size=300)
    config = FoundationConfig(vocab_size=tokenizer.vocab_size, width=32, layers=2, heads=4,
                              kv_heads=2, hidden=64, context=16)
    model = FoundationModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    initial = model.embedding.weight.detach().clone()
    val_sequences = _sequences(validation, tokenizer, config.context)
    baseline = evaluate(model, val_sequences, "cpu")
    (output / "dataset.json").write_text(json.dumps({"format_version": 1, "records": records}, indent=2), encoding="utf-8")
    settings = Settings(_env_file=None, environment="test", database_url=f"sqlite+pysqlite:///{(output / 'jobs.db').as_posix()}",
        supabase_url=None, supabase_publishable_key="", supabase_jwks_url=None, foundation_distributed_training_enabled=True)
    app = create_app(settings)
    Base.metadata.create_all(app.state.engine)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    metrics, workers = [], []
    started = time.perf_counter()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError("Loopback server did not start")
        with httpx.Client(base_url=url, timeout=60) as client:
            credentials = {"email": "distributed-smoke@example.com", "password": secrets.token_urlsafe(32)}
            registration = client.post("/api/v1/auth/register", json=credentials)
            registration.raise_for_status()
            if args.reward_credits:
                # Explicit synthetic issuance in this disposable demo database only.
                ledger = CreditLedger(app.state.engine)
                ledger.create_account("smoke-issuance", "test_issuance", allow_negative=True)
                account = user_account_id(registration.json()["user_id"])
                ledger.create_account(account, "user")
                issued = args.rounds * 2 * args.reward_credits
                ledger.post(idempotency_key="smoke-initial-credits", operation="smoke_issuance",
                            postings={"smoke-issuance": -issued, account: issued})
            response = client.post("/api/v1/auth/login", json=credentials)
            response.raise_for_status()
            client.headers["Authorization"] = "Bearer " + response.json()["access_token"]
            nodes, worker_sessions = [], []
            for i in range(2):
                worker_credentials = {"email": f"worker-{i}@example.com", "password": secrets.token_urlsafe(32)}
                client.post("/api/v1/auth/register", json=worker_credentials).raise_for_status()
                response = client.post("/api/v1/auth/login", json=worker_credentials)
                response.raise_for_status()
                worker_headers = {"Authorization": "Bearer " + response.json()["access_token"]}
                worker_sessions.append(worker_headers)
                response = client.post("/api/v1/nodes", json={"name": f"gradient-worker-{i}",
                    "hardware": detect_hardware().model_dump(mode="json"),
                    "supported_job_types": ["foundation.gradient.v1"], "foundation_torch_version": str(torch.__version__)},
                    headers=worker_headers)
                response.raise_for_status()
                nodes.append(response.json())
            for step in range(1, args.rounds + 1):
                parameters = encode_tensors(dict(model.named_parameters()))
                ids = []
                for shard in range(2):
                    payload = {"shard": {"records": train_records[shard*2:shard*2+2], "tokenizer": tokenizer.to_dict()},
                        "config": asdict(config), "parameters": parameters, "torch_version": str(torch.__version__),
                        "weights_public_and_redistributable": True}
                    response = client.post("/api/v1/jobs/foundation/gradient", json={"payload": payload,
                        "allow_public_node_processing": True, "idempotency_key": f"round-{step}-shard-{shard}",
                        "reward_credits": args.reward_credits})
                    response.raise_for_status()
                    ids.append(response.json()["job_id"])
                for node, worker_device in zip(nodes, args.worker_devices):
                    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
                    worker = subprocess.run([sys.executable, "-m", "ai_home.foundation.node_smoke", "--worker"],
                        input=json.dumps({"url": url, "token": node["node_token"], "allow_foundation_training": True, "foundation_device": worker_device}),
                        capture_output=True, text=True, check=True, timeout=60, env=environment,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    observed = json.loads(worker.stdout)
                    if not observed["completed"]:
                        raise RuntimeError("Worker did not complete a gradient shard")
                    workers.append({"round": step, "node_id": node["node_id"], "device": worker_device, **observed})
                response = client.get("/api/v1/jobs")
                response.raise_for_status()
                jobs = {job["job_id"]: job for job in response.json()}
                if any(jobs[job_id]["status"] != "completed" for job_id in ids):
                    raise RuntimeError("A gradient shard failed verification")
                results = [jobs[job_id]["result"] for job_id in ids]
                update = aggregate_gradients(model, tokenizer, results, optimizer)
                metric = {"round": step, **update, "validation": evaluate(model, val_sequences, "cpu")}
                metrics.append(metric)
                payload = {"format": FORMAT, "config": asdict(config), "tokenizer": tokenizer.to_dict(),
                    "tokenizer_sha256": tokenizer.fingerprint, "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "step": step, "status": "candidate", "device": "cpu",
                    "data_visibility": "public", "training_mode": "verified-gradient-rounds-v1",
                    "metrics": metrics, "validation": metric["validation"], "parameter_count": model.parameter_count,
                    "training_document_hashes": [r["sha256"] for r in prepare_records(train_records)],
                    "validation_document_hashes": [r["sha256"] for r in validation],
                    "dataset_sha256": canonical_hash(prepare_records(train_records)),
                    "gradient_artifact_sha256": [r["artifact_sha256"] for r in results]}
                atomic_save(payload, output / f"round-{step}.pt")
                for i, result in enumerate(results):
                    (output / f"round-{step}-shard-{i}.json").write_text(result["artifact_json"], encoding="utf-8")
            checkpoint = output / f"round-{args.rounds}.pt"
            loaded, loaded_tokenizer, _ = load_checkpoint(checkpoint)
            generated = "".join(stream_text(loaded, loaded_tokenizer, [{"role": "user", "content": "Hola"}], max_new_tokens=16))
            delta = (loaded.embedding.weight.detach() - initial).norm().item()
            if delta <= 0:
                raise RuntimeError("Distributed update did not change parameters")
            report = {"format": "ai-home-distributed-smoke-v1", "torch_version": str(torch.__version__),
                "transport": "loopback HTTP", "device": "cpu", "workers": workers, "metrics": metrics,
                "baseline_validation": baseline, "embedding_delta_l2": delta, "generated_text": generated,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "elapsed_seconds": time.perf_counter() - started, "parameter_count": model.parameter_count,
                "claim": "Real gradient workers and aggregation on one host; no cross-machine performance claim"}
            report["credits"] = {"reward_per_job": args.reward_credits,
                "requester": client.get("/api/v1/credits").json(),
                "workers": [client.get("/api/v1/credits", headers=h).json() for h in worker_sessions],
                "issuance": "Synthetic initial balance in disposable demo database; no money or cryptocurrency"}
            if any(w["earned"] != args.rounds * args.reward_credits for w in report["credits"]["workers"]):
                raise RuntimeError("Verified work was not settled correctly")
            (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        app.state.engine.dispose()


if __name__ == "__main__":
    main()
