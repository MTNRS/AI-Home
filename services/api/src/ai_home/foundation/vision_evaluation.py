"""Greedy heldout caption probes with an explicit white-image control."""

import argparse
import json
from pathlib import Path
import time

import torch

from .data import canonical_hash
from .evaluation import normalize
from .training import select_device
from .vision_checkpoint import load_vision_weights
from .vision_data import load_visual_manifest, load_visual_record
from .vision_inference import stream_vision_tokens


def evaluate_visual(checkpoint, manifest, *, prompt, max_new_tokens=64, device="cpu"):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8192:
        raise ValueError("Invalid visual evaluation instruction")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("Invalid visual evaluation generation budget")
    device = select_device(device)
    records, dataset_hash = load_visual_manifest(manifest)
    model, tokenizer, payload = load_vision_weights(checkpoint, device)
    ids = tokenizer.conversation([{"role": "user", "content": prompt}])
    capacity = model.text.config.context - model.vision_config.patches
    if len(ids) > capacity:
        raise ValueError("Evaluation prompt exceeds visual text context")
    protocol = {"format": "ai-home-visual-evaluation-v1", "prompt": prompt,
        "max_new_tokens": max_new_tokens, "temperature": 0, "control": "all-white-rgb",
        "normalization": "NFC-CRLF-strip-exact", "dataset_sha256": dataset_hash}
    cases = []
    for record in records:
        if record["split"] != "validation":
            continue
        image, expected, _ = load_visual_record(manifest, record, model.vision_config.image_size)
        image = image.to(device)
        outputs = []
        for pixels in (image, torch.ones_like(image)):
            if device == "cuda":
                torch.cuda.synchronize()
            began = time.perf_counter()
            tokens = list(stream_vision_tokens(model, pixels, ids, temperature=0, max_new_tokens=max_new_tokens))
            if device == "cuda":
                torch.cuda.synchronize()
            answer = tokenizer.decode(tokens, errors="replace")
            outputs.append({"answer": answer, "exact_match": normalize(answer) == normalize(expected),
                "generated_tokens": len(tokens), "budget_exhausted": len(tokens) == max_new_tokens,
                "seconds": time.perf_counter() - began})
        cases.append({"image_sha256": record["image_sha256"], "expected": expected,
                      "image": outputs[0], "white_control": outputs[1],
                      "answer_changed": outputs[0]["answer"] != outputs[1]["answer"]})
    count = len(cases)
    return {**protocol, "protocol_sha256": canonical_hash(protocol),
        "checkpoint_sha256": model.checkpoint_sha256, "tokenizer_sha256": tokenizer.fingerprint,
        "device": device, "torch_version": str(torch.__version__), "cases": cases,
        "exact_matches": sum(c["image"]["exact_match"] for c in cases), "count": count,
        "white_control_exact_matches": sum(c["white_control"]["exact_match"] for c in cases),
        "changed_answers": sum(c["answer_changed"] for c in cases),
        "training_dataset_matches": payload["provenance"]["dataset_sha256"] == dataset_hash,
        "claim": "Heldout manifest caption probes only; white-control differences do not prove visual understanding"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = evaluate_visual(args.checkpoint, args.dataset, prompt=args.prompt,
        max_new_tokens=args.max_new_tokens, device=args.device)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=True, indent=2)
    print(json.dumps({k: report[k] for k in ("count", "exact_matches", "white_control_exact_matches", "changed_answers")}))


if __name__ == "__main__":
    main()
