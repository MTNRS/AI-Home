import argparse
from dataclasses import asdict
import json
import signal
import threading
from pathlib import Path

from .data import load_records
from .model import FoundationConfig
from .training import TrainingConfig, TrainingCancelled, train, select_device
from .resources import plan_training


def main():
    parser = argparse.ArgumentParser(description="AI@Home Foundation: training from random weights")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--reviewed-export", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--auto-fit", action="store_true")
    parser.add_argument("--allow-model-shrink", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.allow_model_shrink and not args.auto_fit:
        parser.error("--allow-model-shrink requires --auto-fit")
    if args.auto_fit and args.resume:
        parser.error("Automatic configuration changes are not permitted during resume")
    records = load_records(args.dataset)[0] if args.dataset else None
    model = FoundationConfig(**json.loads(args.model_config.read_text())) if args.model_config else None
    training = TrainingConfig(**json.loads(args.training_config.read_text())) if args.training_config else TrainingConfig()
    if args.reviewed_export and training.objective != "sft":
        parser.error("--reviewed-export requires a training configuration with objective=sft")
    if args.auto_fit:
        args.device = select_device(args.device)
        model, training, plan = plan_training(model or FoundationConfig(), training, args.device,
                                              allow_model_shrink=args.allow_model_shrink)
        plan_path = Path(str(args.output) + ".resource-plan.json")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with plan_path.open("x", encoding="utf-8") as output:
            json.dump(plan, output, indent=2)
        print(json.dumps({"resource_plan": str(plan_path), **plan}), flush=True)
    cancel = threading.Event()
    def interrupt(_signum, _frame):
        if cancel.is_set():
            raise KeyboardInterrupt
        cancel.set()
        print(json.dumps({"status": "stop_requested", "detail": "Finishing the optimizer step and saving a resumable checkpoint"}), flush=True)
    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        result = train(records, args.output, model, training, args.device, args.resume,
                       args.stop_after, on_step=lambda value: print(json.dumps(value), flush=True),
                       reviewed_export=args.reviewed_export, cancel=cancel)
    except TrainingCancelled:
        print(json.dumps({"status": "cancelled_before_update", "checkpoint_created": False}), flush=True)
        raise SystemExit(130)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    print(json.dumps({"checkpoint": str(args.output), "parameters": result["parameter_count"],
                      "validation": result["validation"], "status": "candidate",
                      "training_complete": result["training_complete"], "stop_reason": result["stop_reason"],
                      "training_config": asdict(training)}))


if __name__ == "__main__":
    main()
