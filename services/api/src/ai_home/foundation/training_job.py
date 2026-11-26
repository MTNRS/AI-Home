"""Local training job with a new run directory and retained configuration."""
from dataclasses import asdict
import json
from pathlib import Path

from .data import load_records
from .model import FoundationConfig
from .training import TrainingConfig, train


def read_config(path):
    with Path(path).open("rb") as source:
        raw = source.read(65_537)
    if len(raw) > 65_536:
        raise ValueError("Configuration exceeds input budget")
    return json.loads(raw.decode("utf-8"))


def run_training(dataset, model_config, training_config, output_dir, *, device="cpu",
                 resume=None, cancel=None, on_step=None):
    records, digest = load_records(dataset)
    config = FoundationConfig(**read_config(model_config)) if model_config else None
    settings = TrainingConfig(**read_config(training_config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "ai-home-training-job-v1", "dataset_sha256": digest,
                "model_config": asdict(config) if config else None,
                "training_config": asdict(settings), "device": device,
                "resume": str(Path(resume).resolve()) if resume else None}
    (output_dir / "configuration.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = train(records, output_dir / "model.pt", config, settings, device=device,
                   resume=resume, cancel=cancel, on_step=on_step)
    summary = {"checkpoint": str((output_dir / "model.pt").resolve()), "step": result["step"],
               "training_complete": result["training_complete"], "stop_reason": result["stop_reason"],
               "parameters": result["parameter_count"], "validation": result["validation"],
               "embedding_delta_l2": result["embedding_delta_l2"], "metrics": result["metrics"]}
    (output_dir / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
