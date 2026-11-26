"""Bounded public causal-loss evaluation; scheduler integration is separate."""
import json
import math

from .data import canonical_hash
from .distributed import load_public_task, parameter_fingerprint
from .training import evaluate

EVALUATION_JOB = "foundation.evaluate.v1"


def execute_evaluation(payload, *, device="cpu"):
    validated, model, tokenizer, records, inputs, labels = load_public_task(payload, device=device)
    metrics = evaluate(model, (inputs, labels), device)
    parameters = {key: value.model_dump() for key, value in validated.parameters.items()}
    artifact = {
        "format": "ai-home-evaluation-shard-v1", "objective": "causal",
        "tokens": metrics["tokens"], "mean_loss": metrics["loss"],
        "dataset_sha256": canonical_hash(records),
        "document_hashes": [record["sha256"] for record in records],
        "base_sha256": parameter_fingerprint(validated.config, tokenizer.to_dict(), parameters),
    }
    return {"token_count": metrics["tokens"], "artifact_sha256": canonical_hash(artifact),
            "artifact_json": json.dumps(artifact, sort_keys=True, separators=(",", ":"))}


def verified_evaluation_result(payload, result):
    """Recompute on CPU; never accept a worker's reported loss by itself."""
    try:
        if set(result) != {"token_count", "artifact_sha256", "artifact_json"}:
            return False
        if not isinstance(result["artifact_json"], str) or len(result["artifact_json"].encode()) > 65_536:
            return False
        artifact = json.loads(result["artifact_json"])
        if canonical_hash(artifact) != result["artifact_sha256"]:
            return False
        expected = json.loads(execute_evaluation(payload, device="cpu")["artifact_json"])
        if set(artifact) != set(expected):
            return False
        if type(result["token_count"]) is not int or type(artifact["tokens"]) is not int:
            return False
        if result["token_count"] != expected["tokens"]:
            return False
        if any(artifact[key] != expected[key] for key in expected if key != "mean_loss"):
            return False
        loss = artifact["mean_loss"]
        return (type(loss) in (float, int) and math.isfinite(loss)
                and math.isclose(loss, expected["mean_loss"], rel_tol=1e-5, abs_tol=1e-6))
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return False
