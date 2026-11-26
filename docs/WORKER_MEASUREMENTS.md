# Worker service measurements

`NodeAgent.last_run_metrics` records one completed, server-accepted job using a
monotonic clock. It is cleared at the start of every attempt: idle polls and
failures do not retain a previous job's sample. The record contains job type,
selected device, heartbeat/claim/submit service durations, execution duration,
claim response bytes, submit request bytes, and bytes divided by service time.
Gradient tasks additionally report verified-result token count divided by local
execution time. Node smoke subprocesses include the sample in their JSON output;
distributed smoke reports retain it per worker/round.

These are observations of actual operations, not synthetic hardware scores:

- Heartbeat duration includes HTTP, authentication and database work; it is not
  network-only round-trip latency.
- Claim bytes/time includes scheduler/database delay. Submit bytes/time includes
  server-side independent recomputation and validation, so it is not link upload
  bandwidth. Byte counts cover HTTP bodies, not wire headers/TLS overhead.
- Gradient tokens/second includes checkpoint reconstruction and work execution.
  It is comparable only with equivalent models, tensor shapes, protocols and
  datasets; GPU startup costs can dominate tiny shards.
- Samples contain no prompts, payloads, credentials, lease tokens or account IDs.
  The enclosing experiment report already records synthetic node IDs.

These measurements do not yet change scheduling, credits or verification. The
current scheduler still filters compatibility/memory and uses FIFO. Reporting a
rate is not proof that a volunteer is honest and is not a basis for rewards.
Before scheduling against them, implement bounded/fresh protocol-bound samples,
explicit job constraints and tests for stale/missing/adversarial observations.
Do not infer residential bandwidth from the loopback experiments.

## Observed CPU/CUDA experiment

[Saved real report](WORKER_MEASUREMENTS_SMOKE.json): two rounds with one CPU and
one CUDA worker process per round, actual verified gradients and coordinator
restart reconciliation. CPU execution 2.14–2.20 s (14.53–14.92 tokens/s); CUDA
1.16–1.18 s (27.23–27.52 tokens/s). Heartbeat service times were 7.2–9.3 ms on
loopback. These tiny tasks include initialization and cannot rank hardware for
long training. Neither service byte rate should be described as link bandwidth.
