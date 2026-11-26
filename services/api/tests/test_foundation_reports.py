import json
from pathlib import Path

import pytest

from ai_home.foundation.reports import read_report


DOCS = Path(__file__).parents[3] / "docs"


@pytest.mark.parametrize("filename", ["EVALUATION_BASELINE_CPU.json", "EVALUATION_CONTROL_CPU.json",
                                      "EVALUATION_EXPERIMENT_CPU.json", "EVOLUTION_SMOKE_CPU.json"])
def test_inspect_recorded_real_experiments(filename):
    result = read_report(DOCS / filename)
    assert len(result["file_sha256"]) == 64
    assert result["file_bytes"] == (DOCS / filename).stat().st_size
    if "results" in result["report"]:
        assert len(result["report"]["results"]) == 10


def test_report_rejects_inconsistent_metrics_and_unbounded_input(tmp_path):
    report = json.loads((DOCS / "EVALUATION_BASELINE_CPU.json").read_text())
    report["results"][0]["passed"] = True
    target = tmp_path / "bad.json"
    target.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        read_report(target)
    with pytest.raises(ValueError, match="budget"):
        read_report(target, max_bytes=10)
    for invalid in ['{"format":"ai-home-evaluation-v1","value":NaN}',
                    '{"format":"ai-home-evaluation-v1","value":1e999}',
                    '{"format":"unrecognized"}']:
        target.write_text(invalid)
        with pytest.raises(ValueError):
            read_report(target)
