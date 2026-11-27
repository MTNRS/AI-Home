from dataclasses import replace

import pytest
import torch

from ai_home.foundation.resources import plan_training, estimate_training_bytes
from ai_home.foundation.training import TrainingConfig, train
from test_foundation_model import config, records


def test_planner_preserves_effective_batch_and_trains_real_model(tmp_path):
    torch.set_num_threads(1)
    architecture = config()
    requested = TrainingConfig(steps=1, warmup=0, batch_size=8, accumulation=2)
    budget = estimate_training_bytes(architecture, 2)
    selected_model, selected_training, report = plan_training(
        architecture, requested, "cpu", available_bytes=budget)
    assert selected_model == architecture
    assert selected_training.batch_size == 2
    assert selected_training.accumulation == 8
    assert report["architecture_changed"] is False
    assert report["effective_batch_size"] == 16
    result = train(records(), tmp_path / "planned.pt", selected_model, selected_training, "cpu")
    assert result["embedding_delta_l2"] > 0
    assert (tmp_path / "planned.pt").is_file()


def test_architecture_shrink_requires_explicit_opt_in_and_preserves_tokenizer_context():
    architecture = config()
    requested = TrainingConfig()
    budget = estimate_training_bytes(replace(architecture, layers=1), 1)
    with pytest.raises(ValueError, match="single sample"):
        plan_training(architecture, requested, "cpu", available_bytes=budget)
    selected, training, report = plan_training(architecture, requested, "cpu",
                                              available_bytes=budget, allow_model_shrink=True)
    assert selected.layers < architecture.layers
    assert selected.context == architecture.context
    assert selected.vocab_size == architecture.vocab_size
    assert report["architecture_changed"] is True
    assert estimate_training_bytes(selected, training.batch_size) <= budget
    with pytest.raises(ValueError, match="No architecture"):
        plan_training(architecture, requested, "cpu", available_bytes=1, allow_model_shrink=True)


def test_user_budget_is_respected_even_when_more_hardware_memory_is_available():
    architecture = config()
    budget = estimate_training_bytes(architecture, 1)
    requested = TrainingConfig(batch_size=2, memory_budget_bytes=budget)
    _, selected, report = plan_training(architecture, requested, "cpu", available_bytes=10 * budget)
    assert selected.batch_size == 1
    assert selected.memory_budget_bytes == budget
    assert report["effective_budget_bytes"] == budget
