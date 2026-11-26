import pytest
torch = pytest.importorskip("torch")

from ai_home.foundation.benchmark import benchmark
from ai_home.foundation.model import FoundationConfig


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_compute_benchmark_measures_actual_updates_and_rejects_budget(device):
    config = FoundationConfig(width=32, layers=1, heads=4, kv_heads=2, hidden=64, context=16)
    with pytest.raises(ValueError, match="budget"):
        benchmark(config, device=device, budget_bytes=1)
    result = benchmark(config, device=device, steps=2, warmup=1)
    assert result["max_embedding_change"] > 0
    assert result["tokens_per_second"] == pytest.approx(32 / result["seconds"])
    assert result["sampled_process_rss_bytes"] > 0
    assert len(result["losses"]) == 2
    if device == "cuda":
        assert result["cuda_peak_allocated_bytes"] > result["parameters"] * 4
    else:
        assert result["cuda_peak_allocated_bytes"] is None
