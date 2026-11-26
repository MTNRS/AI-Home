"""Controlled candidate training with replay and a fixed regression holdout."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .checkpoint import load_checkpoint
from .data import load_records
from .training import TrainingCancelled, TrainingConfig, _sequences, evaluate, select_device, train


def compare_checkpoint(path, records, objective, device, max_bytes, *, cache_path=None, cancel=None):
    model, tokenizer, _ = load_checkpoint(path, device=device, max_bytes=max_bytes)
    sequences = _sequences(records, tokenizer, model.config.context, objective, cache_path=cache_path, cancel=cancel)
    return evaluate(model, sequences, device, cancel=cancel)


def run_candidate(parent, output_dir, validation_path, replay_path, *, dataset_path=None,
                  reviewed_export=None, training=None, device="auto", max_loss_increase=0.0,
                  cancel=None, on_step=None, disk_sequences=False):
    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise TrainingCancelled("Candidate comparison cancelled; retained checkpoints are not approved")
    check_cancel()
    if (dataset_path is None) == (reviewed_export is None):
        raise ValueError("Choose exactly one incremental dataset source")
    if not math.isfinite(max_loss_increase) or max_loss_increase < 0:
        raise ValueError("Regression tolerance must be finite and nonnegative")
    training = training or TrainingConfig()
    device = select_device(device)
    validation, validation_hash = load_records(validation_path)
    replay, replay_hash = load_records(replay_path)
    records = load_records(dataset_path)[0] if dataset_path else None
    parent = Path(parent)
    parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
    # Exclusive directory creation prevents replacing an existing experiment.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate = output_dir / "candidate.pt"
    cache_root = output_dir / "sequence-cache" if disk_sequences else None
    payload = train(records, candidate, training=training, device=device,
                    initial_checkpoint=parent, validation_records=validation,
                    replay_records=replay, reviewed_export=reviewed_export, cancel=cancel, on_step=on_step,
                    sequence_cache_dir=cache_root / "training" if cache_root else None)
    check_cancel()
    if hashlib.sha256(parent.read_bytes()).hexdigest() != parent_hash:
        raise RuntimeError("Parent changed during experiment")
    baseline = compare_checkpoint(parent, validation, training.objective, device, training.memory_budget_bytes,
                                  cache_path=cache_root / "baseline" if cache_root else None, cancel=cancel)
    check_cancel()
    result = compare_checkpoint(candidate, validation, training.objective, device, training.memory_budget_bytes,
                                cache_path=cache_root / "candidate" if cache_root else None, cancel=cancel)
    check_cancel()
    delta = result["loss"] - baseline["loss"]
    report = {"format": "ai-home-continual-v1", "status": "candidate",
              "parent_sha256": parent_hash, "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
              "validation_sha256": validation_hash, "replay_sha256": replay_hash,
              "dataset_sha256": payload["dataset_sha256"], "tokenizer_sha256": payload["tokenizer_sha256"],
              "data_visibility": payload["data_visibility"], "training_config": asdict(training),
              "disk_sequences": disk_sequences, "device": device, "baseline": baseline, "candidate": result, "loss_delta": delta,
              "max_loss_increase": max_loss_increase, "regression_gate_passed": delta <= max_loss_increase,
              "replay_samples_per_batch": payload["replay_samples_per_batch"],
              "embedding_delta_l2": payload["embedding_delta_l2"],
              "promotion": "not_performed; independent capability evaluation and approval required"}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--dataset", type=Path)
    sources.add_argument("--reviewed-export", type=Path)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--disk-sequences", action="store_true", help="Store local token sequences on disk, including replay and comparison")
    parser.add_argument("--max-loss-increase", type=float, default=0.0)
    args = parser.parse_args()
    training = TrainingConfig(**json.loads(args.training_config.read_text(encoding="utf-8")))
    print(json.dumps(run_candidate(args.parent, args.output_dir, args.validation, args.replay,
          dataset_path=args.dataset, reviewed_export=args.reviewed_export, training=training,
          device=args.device, max_loss_increase=args.max_loss_increase, disk_sequences=args.disk_sequences), indent=2))


if __name__ == "__main__":
    main()
