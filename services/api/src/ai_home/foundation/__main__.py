import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .data import load_records
from .model import FoundationConfig
from .training import TrainingConfig, train


def main():
    parser = argparse.ArgumentParser(description="AI@Home Foundation: training from random weights")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--reviewed-export", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    records = load_records(args.dataset)[0] if args.dataset else None
    model = FoundationConfig(**json.loads(args.model_config.read_text())) if args.model_config else None
    training = TrainingConfig(**json.loads(args.training_config.read_text())) if args.training_config else TrainingConfig()
    if args.reviewed_export and training.objective != "sft":
        parser.error("--reviewed-export requires a training configuration with objective=sft")
    result = train(records, args.output, model, training, args.device, args.resume,
                   args.stop_after, on_step=lambda value: print(json.dumps(value), flush=True),
                   reviewed_export=args.reviewed_export)
    print(json.dumps({"checkpoint": str(args.output), "parameters": result["parameter_count"],
                      "validation": result["validation"], "status": "candidate",
                      "training_config": asdict(training)}))


if __name__ == "__main__":
    main()
