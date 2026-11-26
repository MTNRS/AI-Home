# Foundation compute measurements

Real FP32 forward/backward, gradient clipping and AdamW updates were measured on
2026-11-26, using the own random-initialized architecture. Configurations are in
`configs/foundation/benchmarks/`. Each run used a fresh process, batch size 1,
context 256, vocabulary 512, two warmup updates and ten measured updates. The
same random-token batch was repeated. These are compute probes, not pretraining,
authorized knowledge acquisition, convergence evidence or language-quality tests.

| Profile | Parameters | Device | Tokens/s | Peak allocated GPU MiB | Peak reserved GPU MiB |
| --- | ---: | --- | ---: | ---: | ---: |
| Nano | 803,968 | CPU, 8 torch threads | 14,961 | — | — |
| Nano | 803,968 | RTX 4070 SUPER | 13,790 | 38.9 | 44 |
| Micro | 4,361,472 | RTX 4070 SUPER | 8,388 | 101.9 | 114 |
| Small | 22,815,232 | RTX 4070 SUPER | 5,762 | 466.7 | 492 |

The profiles increase width/depth while preserving the same task shape, vocabulary
and precision. The measured allocations justify their use as initial local
experiments within the 12 GB GPU, not as optimal sizes. Small's short-run cost is
about 2.4 times Nano's per token while offering about 28 times its parameters.
There is considerable memory headroom in this specific shape; it is deliberately
not filled with unvalidated larger models. Larger vocabularies, contexts, batches,
mixed precision and gradient accumulation require their own measurements.
Medium and Large have no selected parameter counts: multi-GPU infrastructure and
network-aware training measurements are still required before assigning them.

Nano CPU beats CUDA in this short sample, where Python/kernel launch overhead can
matter. Do not generalize that ordering to larger models or sustained training.
The report includes synchronized CUDA timing, PyTorch/device details, sampled RSS,
losses, memory estimates and a measured nonzero weight change. No checkpoint is
published by this benchmark. CUDA allocation excludes memory owned by other
processes and portions of driver/runtime memory; RSS sampling can miss peaks.
Observed process RSS was about 733 MiB for CPU Nano and 1.37–1.45 GiB for CUDA
processes. The planner estimate is a training working-set heuristic, not total
process RSS or an out-of-memory guarantee. It remained above the measured CUDA
allocations in these cases; that does not establish its accuracy for other shapes.

Reproduce with the CUDA-enabled Python environment and `PYTHONPATH=services/api/src`:

```powershell
python -m ai_home.foundation.benchmark configs/foundation/benchmarks/nano.json nano-cpu.json --device cpu --steps 10
python -m ai_home.foundation.benchmark configs/foundation/benchmarks/nano.json nano-cuda.json --device cuda --steps 10
python -m ai_home.foundation.benchmark configs/foundation/benchmarks/micro.json micro-cuda.json --device cuda --steps 10
python -m ai_home.foundation.benchmark configs/foundation/benchmarks/small.json small-cuda.json --device cuda --steps 10
```

Outputs must be new paths. The default 2 GB user budget is additionally capped by
80% of currently available device memory. Runs refuse configurations exceeding
the heuristic before allocating weights; there is no silent model shrinking or
CPU fallback. Raw evidence: `BENCHMARK_NANO_CPU.json`, `BENCHMARK_NANO_CUDA.json`,
`BENCHMARK_MICRO_CUDA.json`, and `BENCHMARK_SMALL_CUDA.json` beside this document.
