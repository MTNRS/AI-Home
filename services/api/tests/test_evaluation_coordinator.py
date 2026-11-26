import copy
import json

import httpx
import pytest

torch = pytest.importorskip("torch")
from ai_home.foundation.checkpoint import atomic_save
from ai_home.foundation.evaluation_coordinator import advance, create_plan
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.training import TrainingConfig, train
from ai_home.node_worker import NodeAgent
from test_foundation_continual import records
from test_job_credits import funded_users
from test_nodes_and_jobs import client


def checkpoint_fixture(tmp_path):
    old, holdout = records("old"), records("holdout")
    for row in old:
        row["split_group_sha256"] = "a" * 64
    config = FoundationConfig(vocab_size=280, width=16, layers=1, heads=2,
                              kv_heads=1, hidden=32, context=16)
    path = tmp_path / "parent.pt"
    state = train(old, path, config, TrainingConfig(steps=1, warmup=0), "cpu", validation_records=holdout)
    return path, state, old, holdout


def test_evaluation_plan_rejects_training_overlap_and_missing_lineage(tmp_path):
    torch.set_num_threads(1)
    path, state, old, holdout = checkpoint_fixture(tmp_path)
    options = dict(server="http://127.0.0.1", owner_user_id="owner", allow_public_node_processing=True,
                   weights_public_and_redistributable=True)
    for shard in (old, [{**r, "split_group_sha256": "a" * 64} for r in holdout]):
        with pytest.raises(ValueError, match="contamination"):
            create_plan(path, [shard], tmp_path / "bad.json", **options)
    with pytest.raises(ValueError, match="duplicate"):
        create_plan(path, [holdout, holdout], tmp_path / "bad.json", **options)
    with pytest.raises(ValueError, match="consent"):
        create_plan(path, [holdout], tmp_path / "bad.json", **{**options, "weights_public_and_redistributable": False})
    for change in ("private", "missing"):
        modified = copy.deepcopy(state)
        if change == "private":
            modified["data_visibility"] = "reviewed-training-only"
        else:
            modified.pop("training_document_hashes")
        altered = tmp_path / f"{change}.pt"
        atomic_save(modified, altered)
        with pytest.raises(ValueError, match="public checkpoint"):
            create_plan(altered, [holdout], tmp_path / "bad.json", **options)
    assert not (tmp_path / "bad.json").exists()


def test_evaluation_plan_resumes_and_aggregates_verified_public_shards(client, tmp_path):
    torch.set_num_threads(1)
    parent, _, _, holdout = checkpoint_fixture(tmp_path)
    owner, _, headers, worker_headers, node_headers = funded_users(client)
    client.app.state.settings.foundation_distributed_evaluation_enabled = True
    plan_path, report_path = tmp_path / "plan.json", tmp_path / "report.json"
    create_plan(parent, [holdout[:2], holdout[2:]], plan_path, server="http://127.0.0.1",
                owner_user_id=owner["user_id"], reward_credits=3,
                allow_public_node_processing=True, weights_public_and_redistributable=True)
    def transport(request):
        response = client.request(request.method, str(request.url), headers=dict(request.headers), content=request.content)
        return httpx.Response(response.status_code, content=response.content)
    coordinator = httpx.Client(base_url="http://127.0.0.1", headers=headers, transport=httpx.MockTransport(transport))
    worker = NodeAgent(server_url="http://127.0.0.1", node_token=node_headers["Authorization"].split()[1],
                       allow_foundation_evaluation=True, transport=httpx.MockTransport(transport))
    try:
        with httpx.Client(base_url="http://127.0.0.1", headers=worker_headers,
                          transport=httpx.MockTransport(transport)) as wrong_owner:
            with pytest.raises(ValueError, match="owner mismatch"):
                advance(wrong_owner, plan_path, parent, report_path)
        assert client.get("/api/v1/jobs", headers=headers).json() == []
        first = advance(coordinator, plan_path, parent, report_path)
        assert first["state"] == "waiting" and len(first["jobs"]) == 2
        assert advance(coordinator, plan_path, parent, report_path) == first
        worker.configure_capabilities()
        assert worker.run_once() and worker.run_once()
        report = advance(coordinator, plan_path, parent, report_path)
        assert report["state"] == "evaluated" and report["promoted"] is False
        from ai_home.foundation.continual import compare_checkpoint
        from ai_home.foundation.data import prepare_records
        reference = compare_checkpoint(parent, prepare_records(holdout), "causal", "cpu", 10_000_000)
        assert report["tokens"] == reference["tokens"]
        assert report["mean_loss"] == pytest.approx(reference["loss"], rel=1e-6, abs=1e-6)
        assert report["tokens"] == sum(s["tokens"] for s in report["shards"])
        assert report["mean_loss"] == sum(s["tokens"] * s["mean_loss"] for s in report["shards"]) / report["tokens"]
        assert advance(coordinator, plan_path, parent, report_path) == report
        assert client.get("/api/v1/credits", headers=worker_headers).json()["earned"] == 6
        assert len(client.get("/api/v1/jobs", headers=headers).json()) == 2
        report_path.write_text("changed report", encoding="utf-8")
        with pytest.raises(ValueError, match="report differs"):
            advance(coordinator, plan_path, parent, report_path)
    finally:
        coordinator.close()
        worker.close()


def test_evaluation_cli_creates_real_plan_and_requires_credentials(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    torch.set_num_threads(1)
    parent, _, _, holdout = checkpoint_fixture(tmp_path)
    shards, plan = tmp_path / "shards.json", tmp_path / "cli-plan.json"
    shards.write_text(json.dumps([holdout[:2], holdout[2:]]), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    env.pop("AI_HOME_COORDINATOR_TOKEN", None)
    command = [sys.executable, "-m", "ai_home.foundation.evaluation_coordinator"]
    result = subprocess.run(command + ["plan", "--checkpoint", str(parent), "--shards", str(shards),
        "--output", str(plan), "--server", "http://127.0.0.1", "--owner", "test-owner",
        "--allow-public-node-processing", "--weights-public-and-redistributable"],
        env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    stored = json.loads(plan.read_text(encoding="utf-8"))
    assert summary == {"state": "planned", "plan_sha256": stored["sha256"], "shards": 2}
    failure = subprocess.run(command + ["advance", "--checkpoint", str(parent), "--plan", str(plan),
        "--output", str(tmp_path / "report.json"), "--server", "http://127.0.0.1"],
        env=env, capture_output=True, text=True, timeout=45)
    assert failure.returncode != 0 and "Set AI_HOME_COORDINATOR_TOKEN" in failure.stderr
    assert not (tmp_path / "report.json").exists()
