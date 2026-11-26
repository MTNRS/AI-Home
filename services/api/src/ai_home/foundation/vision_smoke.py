"""Reproducible authored visual pipeline smoke; not a capability benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

from PIL import Image
import PIL
import torch

from .data import canonical_hash
from .model import FoundationConfig
from .tokenizer import ByteBPE
from .training import TrainingConfig
from .vision import VisionConfig, VisionPrefixModel
from .vision_checkpoint import load_vision_weights
from .vision_data import FORMAT, load_visual_record
from .vision_images import PREPROCESSING
from .vision_inference import stream_vision_text
from .vision_training import train_visual


def run(output_dir, device="cpu"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for index, (color, caption) in enumerate((("red", "red"), ("blue", "blue"),
                                              ("green", "green"), ("yellow", "yellow"))):
        path = output_dir / f"image-{index}.png"
        Image.new("RGB", (8, 8), color).save(path)
        records.append({"image": path.name, "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "caption": caption, "split": "validation" if index == 3 else "train",
            "source": "AI@Home authored solid-color procedural fixture", "url": "local:vision_smoke",
            "license": "Apache-2.0", "date": "2026-11-26", "language": "en",
            "training_allowed": True, "privacy_reviewed": True, "visibility": "public"})
    data = {"format": FORMAT, "preprocessing": PREPROCESSING, "records": records}
    data["dataset_sha256"] = canonical_hash(data)
    manifest = output_dir / "dataset.json"
    manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # Only training captions enter BPE learning, never the validation caption.
    tokenizer = ByteBPE.train([r["caption"] for r in records if r["split"] == "train"], vocab_size=270)
    config = FoundationConfig(vocab_size=tokenizer.vocab_size, width=32, layers=2,
                              heads=4, kv_heads=2, hidden=64, context=32)
    vision = VisionConfig(8, 4)
    training = TrainingConfig(steps=4, batch_size=2, accumulation=1, warmup=1,
                              checkpoint_every=2, objective="sft")
    torch.manual_seed(training.seed)
    initial = VisionPrefixModel(config, vision)
    initial_projection = initial.projection.weight.detach().clone()
    args = (manifest, tokenizer, config, vision, training)
    whole, _, whole_path = train_visual(*args, output_dir / "continuous", device=device, prompt="Q")
    _, _, partial = train_visual(*args, output_dir / "interrupted", device=device, prompt="Q", stop_after=2)
    resumed, history, final = train_visual(*args, output_dir / "resumed", device=device, prompt="Q", resume=partial)
    exact = all(torch.equal(value, resumed.state_dict()[key]) for key, value in whole.state_dict().items())
    delta = (resumed.projection.weight.detach().cpu() - initial_projection).abs().max().item()
    if not exact or not delta > 0:
        raise RuntimeError("Visual smoke failed weight update/resume verification")
    loaded, bpe, _ = load_vision_weights(final, device)
    image, _, _ = load_visual_record(manifest, records[-1], vision.image_size)
    output = "".join(stream_vision_text(loaded, bpe, image.to(device),
        [{"role": "user", "content": "Q"}], temperature=0, max_new_tokens=8))
    report = {"format": "ai-home-visual-smoke-v1", "device": device, "torch": str(torch.__version__),
        "pillow": PIL.__version__, "python": platform.python_version(),
        "dataset_sha256": data["dataset_sha256"], "checkpoint_sha256": loaded.checkpoint_sha256,
        "parameters": sum(p.numel() for p in loaded.parameters()), "exact_resume": exact,
        "visual_max_parameter_delta": delta, "history": history, "generated_text": output,
        "claim": "Technical authored pipeline smoke only; no visual comprehension or generalization demonstrated"}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(run(args.output_dir, args.device), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
