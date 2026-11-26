"""Local training job with a new run directory and retained configuration."""
from dataclasses import asdict
import json
from pathlib import Path

from .data import load_records, canonical_hash
from .contributions import load_reviewed_export
from .model import FoundationConfig
from .training import TrainingConfig, train


def read_config(path):
    with Path(path).open("rb") as source:
        raw = source.read(65_537)
    if len(raw) > 65_536:
        raise ValueError("Configuration exceeds input budget")
    return json.loads(raw.decode("utf-8"))


def run_training(dataset, model_config, training_config, output_dir, *, device="cpu",
                 resume=None, cancel=None, on_step=None, reviewed_export=False,
                 initial_checkpoint=None, validation_dataset=None, disk_sequences=False):
    if resume and initial_checkpoint:
        raise ValueError("Choose exact resume or fresh parent training")
    if reviewed_export:
        reviewed = load_reviewed_export(dataset)
        digest = canonical_hash(reviewed)
        records = None
    else:
        records, digest = load_records(dataset)
    validation = load_records(validation_dataset)[0] if validation_dataset else None
    config = FoundationConfig(**read_config(model_config)) if model_config else None
    settings = TrainingConfig(**read_config(training_config))
    if reviewed_export and settings.objective != "sft":
        raise ValueError("Reviewed exports require the sft training objective")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "ai-home-training-job-v1", "dataset_sha256": digest,
                "model_config": asdict(config) if config else None,
                "training_config": asdict(settings), "device": device,
                "resume": str(Path(resume).resolve()) if resume else None,
                "source_type": "reviewed-training-only" if reviewed_export else "public",
                "initial_checkpoint": str(Path(initial_checkpoint).resolve()) if initial_checkpoint else None,
                "validation_dataset_sha256": canonical_hash(validation) if validation else None,
                "disk_sequences": disk_sequences}
    (output_dir / "configuration.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = train(records, output_dir / "model.pt", config, settings, device=device,
                   resume=resume, cancel=cancel, on_step=on_step,
                   reviewed_export=dataset if reviewed_export else None,
                   initial_checkpoint=initial_checkpoint, validation_records=validation,
                   sequence_cache_dir=output_dir / 'sequence-cache' if disk_sequences else None)
    summary = {"checkpoint": str((output_dir / "model.pt").resolve()), "step": result["step"],
               "training_complete": result["training_complete"], "stop_reason": result["stop_reason"],
               "parameters": result["parameter_count"], "validation": result["validation"],
               "embedding_delta_l2": result["embedding_delta_l2"], "metrics": result["metrics"]}
    (output_dir / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
