import pytest
import torch

from ai_home.foundation.registry import ModelRegistry, file_hash
from ai_home.foundation.training import train, TrainingConfig
from test_foundation_model import config, records
from test_chat_api import client, create_session


def test_real_candidates_approval_activation_rollback_and_api(tmp_path, client):
    torch.set_num_threads(1)
    registry = ModelRegistry(tmp_path / "registry")
    suite = {"version": "fixture-v1", "source": "authored fixture", "license": "Apache-2.0",
             "cases": [{"id": "one", "category": "knowledge", "prompt": "Hi",
                        "answers": ["unreachable expected answer for tiny untrained model"]}]}
    versions = []
    for steps in (1, 2):
        checkpoint = tmp_path / f"model-{steps}.pt"
        train(records(), checkpoint, config(), TrainingConfig(steps=steps, warmup=0), "cpu")
        versions.append(registry.register(checkpoint, suite, max_new_tokens=2))
    with pytest.raises(ValueError, match="No approved"):
        registry.active_checkpoint()
    with pytest.raises(ValueError, match="Only an approved"):
        registry.activate(versions[0], operator="tester", reason="premature")
    with pytest.raises(ValueError, match="gates"):
        registry.approve(versions[0], operator="tester")
    # Explicit zero threshold is only for testing the lifecycle, not a quality claim.
    for version in versions:
        registry.approve(version, operator="fixture-reviewer", minimum_accuracy=0)
    for version in [*versions, versions[0]]:
        registry.activate(version, operator="fixture-operator", reason="lifecycle test")
    assert [r["candidate_id"] for r in registry.history()] == [*versions, versions[0]]
    assert file_hash(registry.active_checkpoint()) == registry.list()[0]["checkpoint_hash"]
    assert len(registry.list()) == 2
    evidence = registry.inspect(versions[0])["evaluation"]
    assert evidence["suite"] == suite
    assert evidence["checkpoint_sha256"] == registry.list()[0]["checkpoint_hash"]
    with pytest.raises(ValueError, match="already recorded"):
        registry.approve(versions[0], operator="another", minimum_accuracy=0)
    assert registry.inspect(versions[0])["approval_comparison"]["result"]["bootstrap"] is True
    comparison = registry.inspect(versions[1])["approval_comparison"]
    assert comparison["baseline_id"] == versions[0]
    assert comparison["result"]["regressions"] == []
    changed_suite = {**suite, "version": "different-suite"}
    incompatible = registry.register(registry.active_checkpoint(), changed_suite, max_new_tokens=2)
    with pytest.raises(ValueError, match="suite_sha256"):
        registry.approve(incompatible, operator="fixture-reviewer", minimum_accuracy=0)
    assert registry.inspect(incompatible)["approved_by"] is None
    assert registry.inspect(incompatible)["approval_comparison"] is None
    token = create_session(client)
    settings = client.app.state.settings
    settings.chat_backend = "foundation"
    settings.foundation_registry = registry.directory
    settings.foundation_device = "cpu"
    settings.foundation_max_new_tokens = 2
    response = client.post("/api/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hi"}]},
        headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"]
    active = registry.active_checkpoint()
    active.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        registry.active_checkpoint()
    response = client.post("/api/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hi"}]},
        headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 503
    assert str(tmp_path) not in response.text


def test_regression_gate_detects_lost_capabilities_despite_equal_total_score():
    from copy import deepcopy
    from ai_home.foundation.registry import compare_evaluations
    baseline = {"suite_sha256": "suite", "checkpoint_sha256": "old",
                "generation": {"temperature": 0, "max_new_tokens": 10},
                "device": "cpu", "torch_version": "test", "all_prompts_fit": True,
                "results": [{"id": "spanish", "category": "spanish", "passed": True},
                            {"id": "math", "category": "mathematics", "passed": False}]}
    candidate = deepcopy(baseline)
    candidate["checkpoint_sha256"] = "new"
    candidate["results"][0]["passed"] = False
    candidate["results"][1]["passed"] = True
    comparison = compare_evaluations(baseline, candidate)
    assert comparison["passed"] is False
    assert comparison["regressions"] == ["spanish"]
    assert comparison["gains"] == ["math"]
    assert comparison["categories"]["spanish"]["delta"] == -1
    assert comparison["categories"]["mathematics"]["delta"] == 1
    candidate["results"][0]["passed"] = True
    assert compare_evaluations(baseline, candidate)["passed"] is True
    for key, value in (("suite_sha256", "other"), ("device", "cuda"),
                       ("torch_version", "different"), ("generation", {}),
                       ("all_prompts_fit", False)):
        changed = {**candidate, key: value}
        with pytest.raises(ValueError):
            compare_evaluations(baseline, changed)
