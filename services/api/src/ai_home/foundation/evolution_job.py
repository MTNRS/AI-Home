"""File-based desktop entry point for a controlled evolution experiment."""
import json
from pathlib import Path

from .evolution import run_experiment
from .training import TrainingConfig
from .training_job import read_config


def run_evolution(parent, suite_path, validation, replay, dataset, training_config, output_dir,
                  *, reviewed_export=False, device="cpu", multiplier=0.5, max_new_tokens=64,
                  cancel=None, on_progress=None, disk_sequences=False):
    with Path(suite_path).open("rb") as source:
        raw = source.read(8_000_001)
    if len(raw) > 8_000_000:
        raise ValueError("Suite exceeds input budget")
    settings = TrainingConfig(**read_config(training_config))
    return run_experiment(parent, output_dir, json.loads(raw.decode("utf-8")), validation, replay,
                          dataset=None if reviewed_export else dataset,
                          reviewed_export=dataset if reviewed_export else None,
                          training=settings, device=device, learning_rate_multiplier=multiplier,
                          max_new_tokens=max_new_tokens, cancel=cancel, on_progress=on_progress,
                          disk_sequences=disk_sequences)
