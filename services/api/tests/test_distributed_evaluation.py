import copy
import json

import pytest

torch = pytest.importorskip("torch")
from ai_home.foundation.data import canonical_hash
from ai_home.foundation.distributed import execute_gradient
from ai_home.foundation.distributed_evaluation import execute_evaluation, verified_evaluation_result
from test_foundation_distributed import setup_shards


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_public_loss_task_matches_gradient_loss_without_mutating_inputs(device):
    _, _, _, shards = setup_shards()
    payload = shards[0]
    before = copy.deepcopy(payload)
    rng = torch.get_rng_state().clone()
    result = execute_evaluation(payload, device=device)
    assert payload == before
    assert torch.equal(rng, torch.get_rng_state())
    assert verified_evaluation_result(payload, result)
    artifact = json.loads(result["artifact_json"])
    gradient = json.loads(execute_gradient(payload, device=device)["artifact_json"])
    assert artifact["tokens"] == gradient["tokens"]
    assert artifact["mean_loss"] == pytest.approx(gradient["mean_loss"], rel=1e-5, abs=1e-6)
    assert artifact["base_sha256"] == gradient["base_sha256"]
    assert "gradients" not in artifact
    for field, replacement in (("mean_loss", artifact["mean_loss"] + 1),
                                ("tokens", True), ("base_sha256", "0" * 64)):
        changed = {**artifact, field: replacement}
        tampered = {**result, "artifact_json": json.dumps(changed), "artifact_sha256": canonical_hash(changed)}
        assert not verified_evaluation_result(payload, tampered)
    assert not verified_evaluation_result(shards[1], result)


def test_evaluation_task_rejects_private_and_nonredistributable_inputs():
    _, _, _, shards = setup_shards()
    for field in ("private", "weights"):
        payload = copy.deepcopy(shards[0])
        if field == "private":
            payload["shard"]["records"][0]["visibility"] = "reviewed-training-only"
        else:
            payload["weights_public_and_redistributable"] = False
        with pytest.raises(ValueError):
            execute_evaluation(payload)
