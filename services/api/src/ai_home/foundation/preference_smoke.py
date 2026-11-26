"""Authored synthetic preference demo; no human-conversation or alignment claim."""
import argparse
import json
from pathlib import Path

import torch
from .checkpoint import load_checkpoint
from .data import canonical_hash
from .inference import stream_text
from .model import FoundationConfig
from .preference_training import train_preferences
from .smoke import smoke_records
from .training import TrainingConfig, train


def run(output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    parent = output / "parent.pt"
    corpus = smoke_records()
    (output / "parent-dataset.json").write_text(json.dumps({"format_version": 1, "records": corpus}), encoding="utf-8")
    train(corpus, parent, FoundationConfig(vocab_size=262, width=16, layers=1, heads=2,
        kv_heads=1, hidden=32, context=64), TrainingConfig(steps=2, warmup=0), device)
    records, pairs = [], []
    for index in range(12):
        ids = [f"00000000-0000-0000-0000-{index*2+side+1:012d}" for side in range(2)]
        for identity, answer in zip(ids, ("A", "B")):
            records.append({"contribution_id": identity, "provider": "ai-home-foundation",
                "source_endpoint": "local:authored-synthetic-fixture", "model": "own-synthetic-smoke",
                "consent_version": "training_consent_v1",
                "consent_scope": "conversation_for_aihome_training_candidates", "consent_at_epoch": 1,
                "reviewer_id": "SYNTHETIC-FIXTURE-NOT-A-HUMAN-REVIEW", "reviewed_at_epoch": 2,
                "review_checklist": {"provider_terms_verified": True, "sensitive_data_reviewed": True,
                                     "reuse_rights_verified": True},
                "messages": [{"role": "user", "content": f"Fixture prompt {index}"},
                             {"role": "assistant", "content": answer}]})
        pairs.append({"chosen_contribution_id": ids[0], "rejected_contribution_id": ids[1], "preference_confirmed": True})
    manifest = {"dataset_version": "synthetic-preferences-v1", "records": records}
    export = {**manifest, "manifest_sha256": canonical_hash(manifest), "item_count": len(records),
        "dataset_id": "00000000-0000-0000-0000-000000000100", "created_at": "2026-11-26T00:00:00Z"}
    review = {"format": "ai-home-preference-review-v1", "export_manifest_sha256": export["manifest_sha256"],
        "reviewer_id": "SYNTHETIC-FIXTURE-NOT-A-HUMAN-REVIEW", "reviewed_at_epoch": 3, "pairs": pairs}
    export_path, review_path = output / "synthetic-export.json", output / "synthetic-review.json"
    for path, value in ((export_path, export), (review_path, review)):
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    options = dict(device=device, steps=20, checkpoint_every=5, learning_rate=.003, validation_fraction=.5)
    partial = train_preferences(parent, export_path, review_path, output / "partial", stop_after=10, **options)
    report = train_preferences(parent, export_path, review_path, output / "resumed",
        resume=output / "partial/candidate.pt", **options)
    model, tokenizer, state = load_checkpoint(output / "resumed/candidate.pt", device=device)
    generated = "".join(stream_text(model, tokenizer, [{"role": "user", "content": "Fixture prompt 99"}],
                                  temperature=0, max_new_tokens=16))
    assert partial["step"] == 10 and report["step"] == 20
    assert report["data_visibility"] == "reviewed-training-only" and report["promoted"] is False
    assert state["data_visibility"] == "reviewed-training-only"
    result = {"format": "ai-home-preference-smoke-v1", "device": device, "torch_version": str(torch.__version__),
        "synthetic_fixture": True, "human_feedback": False, "training": report,
        "generated_text_unscored": generated,
        "scope": "Authored A-versus-B pairs with novel numeric prompts; not language quality or human alignment"}
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.device), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
