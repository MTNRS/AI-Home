"""Heuristic, explicit training resource planning; estimates are not benchmarks."""
from dataclasses import asdict, replace

import psutil
import torch

from .model import FoundationModel


def estimate_training_bytes(config, batch_size):
    with torch.device("meta"):
        parameters = FoundationModel(config).parameter_count
    # Runtime/optimizer workspace allowance, calibrated above a measured tiny CUDA run.
    # Larger kernels may still need more; this is not a hard allocation limit.
    return (64 * 1024 * 1024 + parameters * 20
            + batch_size * config.context * config.width * config.layers * 32
            + batch_size * config.heads * config.context ** 2 * config.layers * 4
            + batch_size * config.context * config.vocab_size * 8)


def available_training_bytes(device):
    free = torch.cuda.mem_get_info()[0] if device == "cuda" else psutil.virtual_memory().available
    return int(free * 0.8)


def plan_training(config, training, device, *, allow_model_shrink=False, available_bytes=None):
    if device not in {"cpu", "cuda"}:
        raise ValueError("Resource planning requires a resolved device")
    available = available_training_bytes(device) if available_bytes is None else available_bytes
    if type(available) is not int or available <= 0:
        raise ValueError("Invalid available memory")
    budget = min(training.memory_budget_bytes, available)
    original_config, original_training = config, training
    effective_batch = training.batch_size * training.accumulation
    batches = [n for n in range(training.batch_size, 0, -1) if effective_batch % n == 0]
    while True:
        for batch in batches:
            estimate = estimate_training_bytes(config, batch)
            if estimate <= budget:
                selected = replace(training, batch_size=batch, accumulation=effective_batch // batch,
                                   memory_budget_bytes=budget)
                return config, selected, {
                    "format": "ai-home-resource-plan-v1", "device": device,
                    "available_budget_bytes": available, "effective_budget_bytes": budget,
                    "estimated_training_bytes": estimate, "effective_batch_size": effective_batch,
                    "requested_model": asdict(original_config), "selected_model": asdict(config),
                    "requested_training": asdict(original_training), "selected_training": asdict(selected),
                    "architecture_changed": config != original_config,
                    "claim": "Heuristic estimate with 64 MiB runtime allowance; not measured peak memory or a quality benchmark",
                }
        if not allow_model_shrink:
            raise ValueError("Requested architecture cannot fit even a single sample; increase budget or explicitly allow model shrinking")
        if config.layers > 1:
            config = replace(config, layers=config.layers - 1)
        elif config.width > 2 * config.heads:
            width = config.width - 2 * config.heads
            hidden = max(8, round(config.hidden * width / config.width / 8) * 8)
            config = replace(config, width=width, hidden=hidden)
        else:
            raise ValueError("No architecture within the requested constraints fits the budget")
