"""Bounded local DPO candidates from explicitly reviewed, restricted pairs."""
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch
from .checkpoint import FORMAT, atomic_save, load_checkpoint
from .data import canonical_hash
from .preference_data import load_preference_pairs, split_preference_pairs
from .preference_objective import dpo_loss, response_log_probability
from .resources import available_training_bytes, estimate_training_bytes
from .training import select_device, TrainingCancelled


def train_preferences(parent_path, export_path, review_path, output_dir, *, device="cpu", steps=20,
                      learning_rate=.0003, beta=.1, seed=42, validation_fraction=.2,
                      resume=None, stop_after=None, cancel=None, checkpoint_every=10, on_metric=None):
    if (type(checkpoint_every) is not int or checkpoint_every < 1
            or type(steps) is not int or not 1 <= steps <= 10000 or type(seed) is not int
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (learning_rate, beta))
            or (stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after <= steps))):
        raise ValueError("Invalid preference training settings")
    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise TrainingCancelled("Preference training cancelled")
    check_cancel()
    pairs = load_preference_pairs(export_path, review_path)
    training, validation = split_preference_pairs(pairs, validation_fraction)
    device = select_device(device)
    reference, tokenizer, parent = load_checkpoint(parent_path, max_bytes=100_000_000)
    if any(key not in parent for key in ("training_document_hashes", "validation_document_hashes",
                                        "training_split_groups", "validation_split_groups")):
        raise ValueError("Preference parent requires document and group lineage")
    needed = estimate_training_bytes(reference.config, 2) + reference.parameter_count * 8
    if needed > available_training_bytes(device):
        raise ValueError("Insufficient device memory for policy and reference")
    def lineage(rows):
        hashes, groups = set(), set()
        for row in rows:
            prefix = "\n".join(message["content"] for message in row["messages"])
            hashes.update(hashlib.sha256((prefix + "\n" + row[key]).encode()).hexdigest()
                          for key in ("chosen", "rejected"))
            groups.update([row["split_group_sha256"], *row["source_split_groups"]])
        return hashes, groups
    train_hashes, train_groups = lineage(training)
    val_hashes, val_groups = lineage(validation)
    if (val_hashes.intersection(parent["training_document_hashes"])
            or val_groups.intersection(parent["training_split_groups"])
            or train_hashes.intersection(parent["validation_document_hashes"])
            or train_groups.intersection(parent["validation_split_groups"])):
        raise ValueError("Preference partition overlaps parent training/validation lineage")
    settings = dict(steps=steps, learning_rate=learning_rate, beta=beta, seed=seed,
                    validation_fraction=validation_fraction, device=device, checkpoint_every=checkpoint_every)
    identity = {"parent_sha256": reference.checkpoint_sha256, "pairs_sha256": canonical_hash(pairs),
                "training_sha256": canonical_hash(training), "validation_sha256": canonical_hash(validation),
                "settings": settings, "torch_version": str(torch.__version__)}
    reference.to(device).eval().requires_grad_(False)
    sampler = torch.Generator().manual_seed(seed)
    start, history = 0, []
    if resume is not None:
        policy, resumed_tokenizer, state = load_checkpoint(resume, device=device, max_bytes=100_000_000)
        if (state.get("preference_identity") != identity or state.get("data_visibility") != "reviewed-training-only"
                or resumed_tokenizer.fingerprint != tokenizer.fingerprint
                or policy.config != reference.config):
            raise ValueError("Preference resume identity mismatch")
        optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
        optimizer.load_state_dict(state["optimizer"])
        sampler.set_state(state["sampler_rng"])
        start, history = state["step"], state["metrics"]
    else:
        policy = copy.deepcopy(reference).requires_grad_(True)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    final_step = steps if stop_after is None else stop_after
    if not 0 <= start < final_step:
        raise ValueError("Preference resume must advance at least one step")
    def encode(rows):
        encoded = []
        for row in rows:
            check_cancel()
            prompt = tokenizer.conversation(row["messages"])
            chosen, rejected = [tokenizer.encode(row[key]) + [258] for key in ("chosen", "rejected")]
            if max(len(prompt)+len(chosen)-1, len(prompt)+len(rejected)-1) > policy.config.context:
                raise ValueError("Preference pair exceeds context; no truncation")
            with torch.no_grad():
                c = response_log_probability(reference, prompt, chosen).detach()
                r = response_log_probability(reference, prompt, rejected).detach()
            encoded.append((prompt, chosen, rejected, c, r))
        return encoded
    train_rows, val_rows = encode(training), encode(validation)
    def evaluate(model):
        margins, losses = [], []
        with torch.no_grad():
            for prompt, chosen, rejected, ref_c, ref_r in val_rows:
                check_cancel()
                c = response_log_probability(model, prompt, chosen)
                r = response_log_probability(model, prompt, rejected)
                margins.append((c-r).item())
                losses.append(dpo_loss(c, r, ref_c, ref_r, beta=beta).item())
        return {"pairs": len(margins), "mean_margin": sum(margins)/len(margins),
                "preferred_fraction": sum(m > 0 for m in margins)/len(margins),
                "mean_dpo_loss": sum(losses)/len(losses)}
    baseline = evaluate(reference)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    policy.train()
    step = start
    def snapshot():
        return {"format": FORMAT, "config": asdict(policy.config), "tokenizer": tokenizer.to_dict(),
        "tokenizer_sha256": tokenizer.fingerprint, "model_state": policy.state_dict(),
        "optimizer": optimizer.state_dict(), "sampler_rng": sampler.get_state(), "step": step,
        "metrics": history, "preference_identity": identity, "status": "candidate",
        "data_visibility": "reviewed-training-only", "parameter_count": policy.parameter_count,
        "training_document_hashes": sorted(set(parent["training_document_hashes"]) | train_hashes),
        "validation_document_hashes": sorted(set(parent["validation_document_hashes"]) | val_hashes),
        "training_split_groups": sorted(set(parent["training_split_groups"]) | train_groups),
        "validation_split_groups": sorted(set(parent["validation_split_groups"]) | val_groups)}
    for next_step in range(start+1, final_step+1):
        if cancel is not None and cancel.is_set():
            break
        row = train_rows[torch.randint(len(train_rows), (1,), generator=sampler).item()]
        prompt, chosen, rejected, ref_c, ref_r = row
        optimizer.zero_grad(set_to_none=True)
        loss = dpo_loss(response_log_probability(policy, prompt, chosen),
                        response_log_probability(policy, prompt, rejected), ref_c, ref_r, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in policy.parameters()):
            raise ValueError("Non-finite preference model weights")
        step = next_step
        history.append({"step": step, "dpo_loss": loss.item()})
        if step % checkpoint_every == 0:
            atomic_save(snapshot(), output / f"step-{step}.pt")
        if on_metric is not None:
            on_metric(dict(history[-1]))
        if cancel is not None and cancel.is_set():
            break
    cancelled = cancel is not None and cancel.is_set()
    if step == start:
        raise TrainingCancelled("Preference training cancelled before an update")
    atomic_save(snapshot(), output / "candidate.pt")
    policy.eval()
    try:
        candidate_metrics = None if cancelled else evaluate(policy)
    except TrainingCancelled:
        cancelled, candidate_metrics = True, None
    with (output / "candidate.pt").open("rb") as source:
        candidate_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    report = {"format": "ai-home-preference-training-v1", "identity": identity, "step": step,
        "candidate_checkpoint_sha256": candidate_sha256,
        "cancelled": cancelled, "training_complete": step == steps, "baseline": baseline,
        "candidate": candidate_metrics, "training_pairs": len(training),
        "data_visibility": "reviewed-training-only", "promoted": False,
        "scope": "Reviewed pair likelihood comparison; not a language capability evaluation"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "export", "review", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--beta", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    try:
        report = train_preferences(args.parent, args.export, args.review, args.output_dir,
            device=args.device, steps=args.steps, learning_rate=args.learning_rate, beta=args.beta,
            seed=args.seed, validation_fraction=args.validation_fraction, resume=args.resume,
            stop_after=args.stop_after, checkpoint_every=args.checkpoint_every)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; completed periodic checkpoints remain in the output directory.\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
