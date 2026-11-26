import pytest
import torch

from ai_home.foundation.inspection import inspect_checkpoint
from ai_home.foundation.training import train, TrainingConfig
from ai_home.foundation.checkpoint import atomic_save
from test_foundation_model import config, records


def test_inspection_uses_verified_model_structure_and_retains_real_metrics(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / "model.pt"
    result = train(records(), path, config(), TrainingConfig(steps=2, warmup=0), "cpu")
    report = inspect_checkpoint(path)
    assert report["parameter_count"] == result["parameter_count"]
    assert report["architecture"] == result["config"]
    assert report["metrics"] == result["metrics"]
    assert report["validation"] == result["validation"]
    assert report["embedding_delta_l2"] > 0
    assert report["declared_parameter_count_matches"] is True
    assert "model_state" not in report and "optimizer" not in report
    assert "training_document_hashes" not in report
    result["parameter_count"] = 999999999
    atomic_save(result, path)
    altered = inspect_checkpoint(path)
    assert altered["parameter_count"] == report["parameter_count"]
    assert altered["declared_parameter_count_matches"] is False
    assert altered["checkpoint_sha256"] != report["checkpoint_sha256"]
    with pytest.raises(ValueError, match="budget"):
        inspect_checkpoint(path, max_bytes=1)
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(Exception):
        inspect_checkpoint(path)
