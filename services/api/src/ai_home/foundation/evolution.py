"""Auditable learning-rate experiment; proposes a decision, never deploys it."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

from .continual import run_candidate
from .evaluation import ProbeSuite, evaluate_checkpoint
from .training import TrainingConfig


def probe_regressions(before, after):
    for field in ("suite_sha256", "generation", "device", "torch_version"):
        if field not in before or field not in after or before[field] != after[field]:
            raise ValueError("Cannot compare different or incomplete evaluation protocols")
    def indexed(report):
        rows = report.get("results")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Individual probe evidence required")
        result, categories = {}, {}
        for row in rows:
            if (not isinstance(row.get("id"), str) or row["id"] in result
                    or not isinstance(row.get("category"), str)
                    or type(row.get("passed")) is not bool):
                raise ValueError("Invalid individual probe evidence")
            result[row["id"]] = row
            values = categories.setdefault(row["category"], [0, 0])
            values[0] += row["passed"]
            values[1] += 1
        expected = {key: {"passed": value[0], "total": value[1], "accuracy": value[0] / value[1]}
                    for key, value in categories.items()}
        if report.get("categories") != expected:
            raise ValueError("Category metrics disagree with individual probes")
        return result
    old, new = indexed(before), indexed(after)
    if set(old) != set(new) or any(old[key]["category"] != new[key]["category"] for key in old):
        raise ValueError("Probe identities differ")
    return sorted(key for key in old if old[key]["passed"] and not new[key]["passed"])


def propose_decision(control, experiment, control_loss, experiment_loss, target):
    regressions = probe_regressions(control, experiment)
    if not control["all_prompts_fit"] or not experiment["all_prompts_fit"]:
        return "reject", "Evaluation prompts exceed context budget"
    if not control["training_lineage_available"] or not experiment["training_lineage_available"]:
        return "reject", "Missing training lineage"
    if regressions:
        return "reject", "Previously passing probes regressed: " + ", ".join(regressions)
    before, after = control["categories"], experiment["categories"]
    if target not in before:
        raise ValueError("Evaluation category mismatch")
    if not math.isfinite(control_loss) or not math.isfinite(experiment_loss) or experiment_loss > control_loss:
        return "reject", "Held-out loss did not match the finite control"
    if after[target]["accuracy"] <= before[target]["accuracy"]:
        return "reject", "Targeted capability improvement not demonstrated"
    return "propose_acceptance", "Measured target improvement; independent review still required"


def run_experiment(parent, output_dir, suite, validation, replay, *, dataset=None,
                   reviewed_export=None, training=None, learning_rate_multiplier=0.5,
                   device="auto", max_new_tokens=64):
    if not math.isfinite(learning_rate_multiplier) or not 0.1 <= learning_rate_multiplier <= 4:
        raise ValueError("Learning-rate multiplier must be between 0.1 and 4")
    if learning_rate_multiplier == 1:
        raise ValueError("Experiment must change the learning rate")
    training = training or TrainingConfig()
    suite = ProbeSuite.model_validate(suite).model_dump()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output_dir / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    baseline = evaluate_checkpoint(parent, suite, device=device, max_new_tokens=max_new_tokens,
                                   max_bytes=training.memory_budget_bytes)
    save("suite.json", suite)
    save("baseline-evaluation.json", baseline)
    target = min(baseline["categories"], key=lambda c: (baseline["categories"][c]["accuracy"], c))
    experimental = replace(training, learning_rate=training.learning_rate * learning_rate_multiplier)
    proposal = {"format": "ai-home-evolution-proposal-v1", "target_category": target,
        "limitation": baseline["categories"][target], "parent_sha256": baseline["checkpoint_sha256"],
        "hypothesis": "Changing the learning rate may improve the weakest measured category without regression",
        "change": {"learning_rate": [training.learning_rate, experimental.learning_rate]},
        "control_config": asdict(training), "experimental_config": asdict(experimental),
        "acceptance_rule": "Strict target accuracy gain, no individual probe regression, loss no worse than finite control",
        "publication": "disabled"}
    save("proposal.json", proposal)
    reports, evaluations = {}, {}
    for name, config in (("control", training), ("experiment", experimental)):
        reports[name] = run_candidate(parent, output_dir / name, validation, replay,
            dataset_path=dataset, reviewed_export=reviewed_export, training=config, device=device)
        if reports[name]["parent_sha256"] != baseline["checkpoint_sha256"]:
            raise RuntimeError("Parent changed after baseline evaluation")
        evaluations[name] = evaluate_checkpoint(output_dir / name / "candidate.pt", suite,
            device=device, max_new_tokens=max_new_tokens, max_bytes=config.memory_budget_bytes)
        save(f"{name}-evaluation.json", evaluations[name])
    decision, reason = propose_decision(evaluations["control"], evaluations["experiment"],
        reports["control"]["candidate"]["loss"], reports["experiment"]["candidate"]["loss"], target)
    if not reports["experiment"]["regression_gate_passed"]:
        decision, reason = "reject", "Experimental candidate regressed against the parent holdout"
    parent_regressions = probe_regressions(baseline, evaluations["experiment"])
    if parent_regressions:
        decision, reason = "reject", "Experimental candidate regressed against parent generation probes: " + ", ".join(parent_regressions)
    result = {"format": "ai-home-evolution-result-v1", "proposal": proposal,
              "control": reports["control"], "experiment": reports["experiment"],
              "decision": decision, "reason": reason, "promoted": False,
              "parent_probe_regressions": parent_regressions,
              "control_probe_regressions": probe_regressions(evaluations["control"], evaluations["experiment"]),
              "suite_sha256": baseline["suite_sha256"],
              "control_categories": evaluations["control"]["categories"],
              "experiment_categories": evaluations["experiment"]["categories"]}
    save("result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output-dir", "suite", "validation", "replay", "training-config"):
        parser.add_argument("--" + name, type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--dataset", type=Path)
    sources.add_argument("--reviewed-export", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--learning-rate-multiplier", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.suite.stat().st_size > 8_000_000:
        parser.error("Suite exceeds input budget")
    result = run_experiment(args.parent, args.output_dir, json.loads(args.suite.read_text(encoding="utf-8")),
        args.validation, args.replay, dataset=args.dataset, reviewed_export=args.reviewed_export,
        training=TrainingConfig(**json.loads(args.training_config.read_text(encoding="utf-8"))),
        device=args.device, learning_rate_multiplier=args.learning_rate_multiplier, max_new_tokens=args.max_new_tokens)
    print(json.dumps({"decision": result["decision"], "reason": result["reason"], "promoted": False}))


if __name__ == "__main__":
    main()
