"""Train a local visual candidate from a reviewed manifest and explicit JSON config."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import threading

from .model import FoundationConfig
from .tokenizer import ByteBPE
from .training import TrainingConfig
from .vision import VisionConfig
from .vision_checkpoint import load_vision_weights
from .vision_data import load_visual_manifest
from .vision_training import train_visual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    with args.config.open("rb") as source:
        raw = source.read(65537)
    if len(raw) > 65536:
        raise ValueError("Visual training configuration exceeds 64 KiB")
    data = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(data, dict) or set(data) != {"model", "vision", "training", "prompt", "tokenizer_vocab_size"}:
        raise ValueError("Expected explicit model, vision, training, prompt and tokenizer_vocab_size")
    training = TrainingConfig(**data["training"])
    vision = VisionConfig(**data["vision"])
    records, dataset_hash = load_visual_manifest(args.dataset)
    if "vocab_size" in data["model"]:
        raise ValueError("Use tokenizer_vocab_size; model vocabulary is derived from BPE")
    if args.resume:
        loaded, tokenizer, _ = load_vision_weights(args.resume, max_bytes=training.memory_budget_bytes)
        del loaded
    else:
        tokenizer = ByteBPE.train([r["caption"] for r in records if r["split"] == "train"],
                                  vocab_size=data["tokenizer_vocab_size"])
    config = FoundationConfig(**data["model"], vocab_size=tokenizer.vocab_size)
    # Exclusive output directory prevents overwriting another run's checkpoints.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cancel = threading.Event()
    previous = signal.signal(signal.SIGINT, lambda *_: cancel.set())
    try:
        _, history, checkpoint = train_visual(args.dataset, tokenizer, config, vision, training,
            args.output_dir, device=args.device, prompt=data["prompt"], resume=args.resume,
            stop_after=args.stop_after, cancel=cancel)
    finally:
        signal.signal(signal.SIGINT, previous)
    report = {"format": "ai-home-visual-training-report-v1", "dataset_sha256": dataset_hash,
        "model": asdict(config), "vision": asdict(vision), "training": asdict(training),
        "tokenizer_sha256": tokenizer.fingerprint, "prompt": data["prompt"],
        "checkpoint": checkpoint.name if checkpoint else None, "history": history,
        "cancelled": cancel.is_set(), "status": "candidate", "quality_claim": "none"}
    with (args.output_dir / "training-report.json").open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
