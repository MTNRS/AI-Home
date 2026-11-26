"""Run a local evaluation and exclusively publish its complete report."""
import json
import os
from pathlib import Path
import tempfile

from .evaluation import evaluate_checkpoint, EvaluationCancelled


def run_evaluation(checkpoint, suite_path, output, *, device="cpu", max_new_tokens=64,
                   cancel=None, on_probe=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Evaluation output already exists")
    with Path(suite_path).open("rb") as source:
        raw = source.read(8_000_001)
    if len(raw) > 8_000_000:
        raise ValueError("Suite exceeds input budget")
    report = evaluate_checkpoint(checkpoint, json.loads(raw.decode("utf-8")), device=device,
                                 max_new_tokens=max_new_tokens, cancel=cancel, on_probe=on_probe)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / "report.json"
        with temporary.open("w", encoding="utf-8") as target:
            json.dump(report, target, indent=2, allow_nan=False)
            target.flush()
            os.fsync(target.fileno())
        if cancel is not None and cancel.is_set():
            raise EvaluationCancelled("Evaluation cancelled before report publication")
        os.link(temporary, output)
    return report
