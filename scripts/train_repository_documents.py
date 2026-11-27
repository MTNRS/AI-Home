"""Small, pinned technical-document experiment; no general capability claim."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

import torch
from ai_home.foundation.data import canonical_hash, prepare_records
from ai_home.foundation.checkpoint import load_checkpoint
from ai_home.foundation.inference import stream_text
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.training import TrainingConfig, train

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--revision", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
parser.add_argument("--prepare-only", action="store_true", help="Save the pinned datasets/configuration without training")
args = parser.parse_args()
if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
    parser.error("Use an immutable full commit SHA")
root = Path(__file__).resolve().parents[1]
output = args.output_dir.resolve()
output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2)
sources = [("docs/FOUNDATION_MODEL.md", 100), ("docs/TOKENIZER.md", 85),
           ("docs/TRAINING.md", 75), ("docs/EVALUATION.md", 75)]
records = []
for path, line_limit in sources:
    raw = subprocess.run(["git", "show", f"{args.revision}:{path}"], cwd=root,
                         check=True, capture_output=True).stdout
    text = "\n".join(raw.decode("utf-8").splitlines()[:line_limit])
    records.append({"source": f"AI@Home project documentation: {path}",
        "url": f"https://github.com/MTNRS/AI-Home/blob/{args.revision}/{path}",
        "license": "Apache-2.0", "date": "2026-11-26", "language": "es-en",
        "text": text, "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_path": path, "source_commit": args.revision, "excerpt_first_lines": line_limit,
        "split_group_sha256": canonical_hash({"family": "github-file-v1", "repository": "mtnrs/ai-home", "path": path}),
        "training_allowed": True, "privacy_reviewed": True, "visibility": "public"})
training_records, validation = prepare_records(records[:2]), prepare_records(records[2:])
for name, rows in (("training", training_records), ("validation", validation)):
    (output / f"{name}.json").write_text(json.dumps({"format_version": 1,
        "dataset_version": canonical_hash(rows), "records": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
config_path = "configs/foundation/benchmarks/nano.json"
config_raw = subprocess.run(["git", "show", f"{args.revision}:{config_path}"], cwd=root,
                            check=True, capture_output=True).stdout
config = FoundationConfig(**json.loads(config_raw))
settings = TrainingConfig(steps=200, warmup=10, checkpoint_every=50,
                          precision="bf16" if args.device == "cuda" else "fp32")
(output / "configuration.json").write_text(json.dumps({"model": asdict(config), "training": asdict(settings),
    "model_config_path": config_path, "model_config_source_sha256": hashlib.sha256(config_raw).hexdigest(),
    "source_revision": args.revision, "split": "first two files train; other two files validation",
    "scope": "Reviewed project documentation excerpts, some historical statements; not a general corpus"}, indent=2), encoding="utf-8")
if args.prepare_only:
    print(json.dumps({"state": "prepared", "training_documents": len(training_records),
                      "validation_documents": len(validation), "source_revision": args.revision}))
    raise SystemExit(0)
baseline = output / "one-step.pt"
initial = train(training_records, baseline, config, settings, args.device, stop_after=1, validation_records=validation)
candidate = output / "candidate.pt"
final = train(training_records, candidate, config, settings, args.device, resume=baseline, validation_records=validation)
model, tokenizer, state = load_checkpoint(candidate, device=args.device)
generated = "".join(stream_text(model, tokenizer, [{"role": "user", "content": "Explica el tokenizador de AI@Home."}],
                              temperature=0, max_new_tokens=64))
report = {"format": "ai-home-repository-document-experiment-v1", "source_revision": args.revision,
    "device": args.device, "torch_version": str(torch.__version__), "parameter_count": final["parameter_count"],
    "step": final["step"], "one_step_validation": initial["validation"], "candidate_validation": final["validation"],
    "embedding_delta_l2_since_step_one": final["embedding_delta_l2"], "dataset_sha256": final["dataset_sha256"],
    "tokenizer_sha256": tokenizer.fingerprint, "checkpoint_sha256": model.checkpoint_sha256,
    "generated_text": generated, "metrics": final["metrics"], "promoted": False,
    "scope": "Four technical files; related topics across files; no independent language capability benchmark"}
(output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "metrics"}, ensure_ascii=True, indent=2))
