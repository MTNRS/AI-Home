"""Reproduce the small, authored-fixture continual-learning experiment."""
import argparse
from dataclasses import asdict
import json

import torch

from .continual import run_candidate
from .model import FoundationConfig
from .training import TrainingConfig, train


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--context", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    datasets = {}
    for split, prefix in (("base", "Primer corpus"), ("incremental", "Nuevos ejemplos"),
                          ("validation", "Evaluacion reservada")):
        datasets[split] = [{"source": "AI@Home authored continual smoke fixture",
            "url": "local:ai_home.foundation.continual_smoke", "license": "Apache-2.0",
            "date": "2026-11-26", "language": "es", "training_allowed": True,
            "privacy_reviewed": True, "visibility": "public",
            "text": f"{prefix} {i}: Un modelo propio aprende con datos autorizados. " * 8}
            for i in range(4)]
        (args.output_dir / f"{split}.json").write_text(json.dumps(
            {"format_version": 1, "records": datasets[split]}, indent=2), encoding="utf-8")
    config = FoundationConfig(vocab_size=280, width=32, layers=2, heads=4,
                              kv_heads=2, hidden=64, context=args.context)
    settings = TrainingConfig(steps=8, warmup=2, batch_size=2, checkpoint_every=4)
    for name, value in (("model-config", config), ("training-config", settings)):
        (args.output_dir / f"{name}.json").write_text(json.dumps(asdict(value), indent=2), encoding="utf-8")
    parent = args.output_dir / "parent.pt"
    train(datasets["base"], parent, config, settings, args.device,
          validation_records=datasets["validation"])
    report = run_candidate(parent, args.output_dir / "candidate-run",
        args.output_dir / "validation.json", args.output_dir / "base.json",
        dataset_path=args.output_dir / "incremental.json", training=settings, device=args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
