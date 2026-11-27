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

## Coordinator CLI

The `plan` command accepts `--scheduling requirements.json`; the input is limited
to 64 KiB and validated before loading the parent checkpoint or publishing a plan.
UTF-8 with or without a BOM is accepted. Output displays the effective constraints
beside the plan hash. Omitting the option preserves unconstrained plans.

```powershell
python -m ai_home.foundation.coordinator plan parent.pt shards.json plan.json --server https://your-server.example --owner YOUR_USER_ID --allow-public-node-processing --scheduling configs/foundation/scheduling/loopback-example.json
```

The example file contains the permissive thresholds used for the local smoke,
not recommended thresholds for arbitrary networks or hardware. Choose explicit
limits from comparable observations and keep bootstrap jobs available. Planning
is local: this command does not authenticate or submit jobs. `advance` uses the
existing `AI_HOME_COORDINATOR_TOKEN` environment variable and sends the immutable
plan requirements on each idempotent submission. Changing the requirements means
creating a different plan; it does not amend or cancel the original funded jobs.

Validation additionally invoked the real CLI against the previously trained
round-1 checkpoint and matching round-2 shards. It produced a new bounded plan
without contacting a service. API tests separately verified that CLI-created
requirements reach both jobs and are retained across retries. Those tests also
check BOM input, malformed/empty requirements, and oversized-file rejection
without changing the original published plan.

## Desktop job summaries

The node page now includes **Mis trabajos distribuidos**. The authenticated
read-only window lists jobs owned by the current account in pages of 50,
with status/type/ID, creation time, attempts, failure code and scheduling limits.
It does not display prompts, gradient artifacts, payloads, lease credentials or
other accounts' jobs. `GET /api/v1/jobs/summary` projects only summary columns in
the database; existing full-result job endpoints remain available to their owners.
The desktop helper bounds the response to 2 MB and validates the typed summaries.

Refresh is manual, disables duplicate requests and marks previously displayed
information as potentially stale while loading or after failure. Session/window
validity guards discard late callbacks; the private window closes with profile
changes through the existing account isolation mechanism. A queued state is not
a diagnosed cause: the dialog explicitly explains that it does not confirm node
availability and that bootstrap work may be needed for new nodes.

Tests cover authentication/account isolation, omitted payload/result fields,
HTTP-helper parsing/limits and constraint rendering. A hidden real-Tk functional
smoke verified list population, selection, displayed requirements and read-only
text. This functional test is separate from the later [visual review](DESKTOP_VISUAL_REVIEW.md); DPI/accessibility coverage remains incomplete. The view does not yet
create/cancel jobs or diagnose all reasons for waiting.


## Stable summary pagination

The summary API keeps its default response shape/list and 100-row default limit.
Optional `limit` (1–100) and the paired `before_created_at` / `before_job_id` cursor
select older records by descending creation timestamp and UUID. Both cursor
fields are required together; invalid datetime/UUID/limit values return 422.
Every page independently filters the authenticated owner. A cursor is a sorting
boundary, not proof of ownership or a resource authorization token.

The desktop requests 51 rows to display 50 and detect another page, with Inicio,
Más recientes and Más antiguos controls. History advances only after a successful
response; failures keep the previous page and cursor. Inicio retrieves the latest
page, including newly inserted jobs. This is a live view, not a transaction-wide
snapshot: statuses/deletions may change between requests. Newer insertions do not
shift the older-page boundary as offset pagination would.

SQLite CURRENT_TIMESTAMP stores whole seconds without a fractional suffix while
SQLAlchemy binds six-digit fractions. The query normalizes SQLite date strings
to six digits for both ordering and cursor comparison, preserving microseconds;
PostgreSQL compares its native timestamps. This avoids repeated tied pages.
SQLite normalization is a correctness measure, not a large-scale query benchmark.

Verified 105 jobs across tied timestamps, mixed zero/nonzero microsecond storage,
a newer insert between pages, no duplicates/omissions, invalid cursor pairs and
limits. A PostgreSQL integration test separately exercised native tied timestamps.
Hidden real-Tk widgets verified forward/back navigation, latest-page return and
preservation of the old page on failed navigation. A subsequent [visual review](DESKTOP_VISUAL_REVIEW.md) checked default/minimum sizes; other DPI settings remain unverified.
