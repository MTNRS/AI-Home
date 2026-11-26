# Optional measured gradient scheduling

Gradient job creation accepts optional `scheduling` requirements:

```json
{
  "min_gradient_tokens_per_second": 1.0,
  "min_claim_service_bytes_per_second": 1000.0,
  "min_submit_service_bytes_per_second": 1000.0,
  "max_heartbeat_service_seconds": 1.0
}
```

At least one positive finite bounded threshold is required. The server adds a
profile SHA256 covering the validated architecture, complete tokenizer and
PyTorch version. Changing weights between rounds does not change this profile;
changing architecture/context/tokenizer/runtime does. Constraints are returned
in job responses and included in owner-scoped idempotency hashes. Changing a
constraint under the same idempotency key returns 409. Existing unconstrained
jobs retain their old hashes and selection behavior.

At claim time the scheduler requires both the resource snapshot and completed-job
sample to be at most 120 seconds old, a matching selected CPU/CUDA device, a
matching profile, and all requested thresholds. Missing/old/pre-profile samples
cannot satisfy constrained jobs. Compatibility, consent, available RAM/VRAM,
runtime and lease checks still apply. The database query filters eligibility
before FIFO ordering and `FOR UPDATE SKIP LOCKED`; an ineligible early job cannot
block a later eligible job. There is no silent relaxation or timed fallback.

New nodes need unconstrained bootstrap work to obtain a verified completion and
upload a sample. A queue containing only constrained jobs can legitimately wait
forever without eligible observations. Owners must plan bootstrap capacity or
cancel/recreate requirements; no extra reward is paid for reported performance.
The coordinator Python `create_plan(..., scheduling=...)` binds constraints into
its immutable plan/hash and resends them unchanged on restart.

These are worker-reported **service** observations, not hardware attestation or
measured physical network bandwidth. A dishonest worker can exaggerate times or
bytes; verified result recomputation remains the credit gate. Comparable model
configuration reduces mismatch but does not normalize cache, startup, host load,
shard length or data shape. The policy filters work for a polling node; it is not
a global optimal assignment or a demonstrated throughput improvement.

Apply migration `0010_job_scheduling` before deploying the updated server:
`python -m ai_home.migrate upgrade`. It adds nullable `compute_jobs.scheduling`;
existing rows remain unconstrained. No production database was modified here.

Validation:

- SQLite API tests exercise missing samples, all four failing thresholds, matching
  constraints, idempotency/conflict, device/profile changes and sample expiry.
  They execute a real verified gradient to obtain the receipt, then use controlled
  timing fixtures to test decisions; those fixture rates are not benchmarks.
- PostgreSQL 17 local: all six integration tests passed, including migration/schema
  match and the actual JSON eligibility query for matching and expired samples.
- Real CPU/CUDA run: `python -m ai_home.foundation.distributed_smoke --output-dir
  artifacts/scheduling-v1 --rounds 2 --reward-credits 3 --worker-devices cpu cuda
  --restart-coordinator --measured-scheduling`. Two initial jobs bootstrap samples;
  two second-round jobs have all four requirements. All completed, all four
  observations uploaded, coordinator retries preserved work and credits settled.
  [Saved report](MEASURED_SCHEDULING_SMOKE.json). This is loopback on one host.
- Full local suite: 296 passed, six optional PostgreSQL cases skipped; the separate
  PostgreSQL invocation above executed those six successfully.
