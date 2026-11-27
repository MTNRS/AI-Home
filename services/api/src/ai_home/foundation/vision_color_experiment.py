"""Authored color-learning experiment; no claim about real-world vision."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch

from .data import canonical_hash
from .model import FoundationConfig
from .tokenizer import ByteBPE
from .training import TrainingConfig
from .vision import VisionConfig, VisionPrefixModel
from .vision_checkpoint import save_vision_weights
from .vision_data import FORMAT
from .vision_images import PREPROCESSING
from .vision_training import train_visual
from .vision_evaluation import evaluate_visual


def run(output_dir, device="cpu"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    # Disjoint intensity values, same three classes. No photos or private inputs.
    for channel, label in enumerate(("red", "green", "blue")):
        for intensity in (140, 160, 180, 200, 220, 240, 150, 210):
            rgb = [0, 0, 0]
            rgb[channel] = intensity
            path = output / f"{label}-{intensity}.png"
            Image.new("RGB", (8, 8), tuple(rgb)).save(path)
            records.append({"image": path.name, "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "caption": label, "split": "validation" if intensity in (150, 210) else "train",
                "source": "AI@Home authored RGB intensity experiment", "url": "local:vision_color_experiment",
                "license": "Apache-2.0", "date": "2026-11-26", "language": "en",
                "training_allowed": True, "privacy_reviewed": True, "visibility": "public"})
    data = {"format": FORMAT, "preprocessing": PREPROCESSING, "records": records}
    data["dataset_sha256"] = canonical_hash(data)
    manifest = output / "dataset.json"
    manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tokenizer = ByteBPE.train([r["caption"] for r in records if r["split"] == "train"], vocab_size=270)
    config = FoundationConfig(vocab_size=tokenizer.vocab_size, width=32, layers=2,
                              heads=4, kv_heads=2, hidden=64, context=16)
    vision = VisionConfig(8, 4)
    training = TrainingConfig(steps=200, batch_size=4, warmup=10, learning_rate=.003,
                              checkpoint_every=100, objective="sft")
    torch.manual_seed(training.seed)
    initial = VisionPrefixModel(config, vision)
    parent = output / "initial.pt"
    save_vision_weights(initial, tokenizer, parent, provenance={"dataset_sha256": data["dataset_sha256"],
        "visibility": "public", "training_allowed": True, "privacy_reviewed": True})
    before = evaluate_visual(parent, manifest, prompt="Color?", max_new_tokens=8, device=device)
    _, _, candidate = train_visual(manifest, tokenizer, config, vision, training, output / "training",
                                   device=device, prompt="Color?")
    after = evaluate_visual(candidate, manifest, prompt="Color?", max_new_tokens=8, device=device)
    report = {"format": "ai-home-color-experiment-v1", "before": before, "after": after,
        "claim": "Only synthetic single-channel solid colors at heldout intensities; not natural-image understanding"}
    (output / "experiment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"before": before["exact_matches"], "after": after["exact_matches"],
            "count": after["count"], "white_control": after["white_control_exact_matches"],
            "changed_answers": after["changed_answers"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(run(args.output_dir, args.device)))


if __name__ == "__main__":
    main()
