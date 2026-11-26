import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .inference import stream_text
from .model import FoundationConfig
from .training import TrainingConfig, train


def smoke_records():
    """Redistributable authored fixture; deliberately not a capability corpus."""
    records = []
    for i, subject in enumerate(("datos", "modelo", "memoria", "nodos", "pruebas", "código", "tokens", "contexto")):
        records.append({"source": "AI@Home authored synthetic smoke fixture",
                        "url": "local:ai_home.foundation.smoke", "license": "Apache-2.0",
                        "date": "2026-11-26", "language": "es", "training_allowed": True,
                        "privacy_reviewed": True, "visibility": "public",
                        "text": (f"Ejemplo {i}: {subject}. AI@Home aprende mediante entrenamiento. " * 16)})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    records = smoke_records()
    dataset = args.output_dir / "dataset.json"
    dataset.write_text(json.dumps({"format_version": 1, "records": records}, ensure_ascii=False), encoding="utf-8")
    config = FoundationConfig(vocab_size=288, width=32, layers=2, heads=4,
                              kv_heads=2, hidden=64, context=16)
    settings = TrainingConfig(steps=8, warmup=2, batch_size=2, checkpoint_every=4)
    path = args.output_dir / "foundation-nano.pt"
    train(records, path, config, settings, args.device, stop_after=4)
    result = train(records, path, config, settings, args.device, resume=path)
    model, tokenizer, loaded = load_checkpoint(path, device=result["device"])
    generated = "".join(stream_text(model, tokenizer, [{"role": "user", "content": "Hola"}],
                                  max_new_tokens=32, temperature=0.8))
    if result["embedding_delta_l2"] <= 0:
        raise RuntimeError("Training did not change embedding parameters")
    report = {"format": loaded["format"], "device": result["device"],
              "inference_device": str(next(model.parameters()).device),
              "python_version": platform.python_version(), "torch_version": str(torch.__version__),
              "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "parameter_count": result["parameter_count"], "step": result["step"],
              "tokenizer_sha256": tokenizer.fingerprint, "dataset_sha256": result["dataset_sha256"],
              "embedding_delta_l2_since_resume": result["embedding_delta_l2"],
              "validation": result["validation"], "metrics": result["metrics"],
              "generated_text": generated,
              "claim": "Mechanical smoke experiment only; no conversational capability established"}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows consoles can use cp1252; generated Unicode must not fail the run.
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
