# Experimental own-model low-rank updates

`foundation.adapters` implements a small q/v low-rank update primitive for the
own Foundation architecture, following the frozen-base factorization described
in [Hu et al., LoRA](https://arxiv.org/abs/2106.09685). No third-party model weights
or adapter package are used.

`attach_adapters` copies the input model, freezes its original parameters and
adds trainable A/B factors to query and value projections. A uses a private seeded
CPU generator; B starts at zero, so initial outputs equal the original model.
Ranks must fit both projection dimensions. Repeated attachment is rejected.
`merge_adapters` creates another copy with the updates folded into ordinary
linear weights, retaining the standard Foundation architecture and KV-cache
behavior. Neither copy retains a stale loaded-checkpoint identity.

Three tests passed with actual CPU/CUDA optimization. The tiny fixture trains
112 adapter parameters for twelve AdamW updates; output changes while frozen
parameters and the original model remain bitwise unchanged. Merged outputs and
cached inference match within floating-point tolerance. This is a numerical
primitive, not an adapter training workflow or a language-quality result.

Still required: authorized dataset integration,
optimizer resume, lineage and restricted/public visibility propagation, resource
measurements, independent evaluation and a typed node protocol. This module does
not submit distributed work, save a publishable candidate or authorize data use.

## Restricted factor artifacts

`save_adapter(model, parent_path, output_path)` now checks the parent architecture
and every frozen tensor before writing factors only. It binds the artifact to the
loaded parent-file SHA256 and tokenizer fingerprint. Output is exclusive and
always marked `reviewed-training-only`/`candidate`; this is a conservative label,
not proof of authorized training data. The parent budget is 100 MB and adapter
file budget 16 MB. These file limits are not hard process-memory guarantees.

`load_adapter` uses the bounded `weights_only=True` reader and checks exact schema,
parent identity, target names, shapes, dtypes and finite factors before loading.
It returns an adapted model without a stale parent-checkpoint identity. The
format does not yet contain optimizer state or training-data lineage, and must
not be used as a distributed training or publication artifact.

Four tests passed in 7.54 s, including the CPU/CUDA numerical tests and a factor
round trip. The artifact test verifies exact restored tensors, refusal to
overwrite, rejection of another parent file even with identical model tensors,
and refusal to save after modifying the frozen base. No full model weights or
third-party adapters are stored in the factor artifact.

## Authored-data CPU/CUDA experiment

Run `python -m ai_home.foundation.adapter_smoke --output-dir NEW_DIR --device cuda`
(or `cpu`). The command trains a tiny own parent from random initialization,
reuses the existing authorized-record partition, sequence preparation and loss
evaluation functions, and trains q/v factors for forty actual AdamW steps. It
saves/reloads the factor artifact and generates through a merged model. The
original parent's tensors are checked unchanged. This is a fixed synthetic
experiment, not the general adapter trainer.

CPU and CUDA runs both exited zero. Base: 18,528 parameters; trainable factors:
448; saved factor file: 4,381 bytes. On 788 reserved tokens, loss moved from
approximately 5.634474 to 5.585938. Greedy generation repeats `ent`; no language
quality is demonstrated. Reports: [CPU](ADAPTER_SMOKE_CPU.json) and
[CUDA](ADAPTER_SMOKE_CUDA.json). Full datasets, parent checkpoints and adapters
remain under `artifacts/adapter-smoke-{cpu,cuda}-v1`. The file size excludes the
required parent checkpoint; it is not a measured network or memory speedup.

## Optional cumulative data lineage

Saving can now take `training_records` and `validation_records` together. Both
pass the existing public-data authorization/provenance/privacy checks. The
artifact retains a dataset fingerprint and the union of parent/new document and
split-family fingerprints; overlapping training/validation history is rejected.
Loading validates those fingerprints and requires that no parent history was
omitted. Artifacts without lineage remain readable as numerical artifacts, with
no implication that their training data were validated. Record assertions are
not independently authenticated, and this API does not accept private exports.

The updated CPU smoke completed forty steps and exact reload with lineage at
`artifacts/adapter-lineage-cpu-v1`. Its factor artifact is 5,213 bytes; the earlier
4,381-byte reports predate this metadata. Authorization denial and contamination
against parent training history are covered by the adapter tests. General trainer
integration, optimizer resume and a merged checkpoint with propagated lineage
remain pending.

## Merged Foundation checkpoint export

`export_merged_adapter(parent_path, adapter_path, output_path)` now requires
validated cumulative lineage, loads the bound parent and factors, and writes a
new ordinary Foundation checkpoint exclusively. It preserves tokenizer, dataset
and cumulative split fingerprints, parent/adapter digests, restricted visibility
and candidate status. It does not attach an optimizer or activate the model;
this is an inference/evaluation candidate, not an exact-resume checkpoint.

The forty-step CPU artifact was exported to
`artifacts/adapter-lineage-cpu-v1/merged.pt`; loading through the standard loader
verified every tensor against the in-memory merge and every lineage field against
the adapter. Six adapter tests passed in 5.35 s, including refusal to export
without lineage, refusal to overwrite, standard-loader equivalence and rejection
of a known training document used as an evaluation probe. The general trainer and
optimizer resume remain pending.

## Public-data trainer and exact resume

`foundation.adapter_training` now accepts a bounded authorized public dataset,
uses the existing partition and sequence functions, checks cumulative lineage,
and runs local FP32 causal AdamW training. It uses a conservative memory estimate,
gradient clipping and finite-update checks. It does not accept private exports,
train on volunteer nodes, perform mixed precision or promote candidates.

```powershell
python -m ai_home.foundation.adapter_training --parent parent.pt --dataset data.json --output-dir new-run --device cpu --steps 20 --checkpoint-every 10
```

Each completed `step-N` directory contains restricted `adapter.pt` and
`training.pt` with AdamW, sampler state, history, artifact hash and input/runtime
identity. Use `--stop-after 10`, then `--resume new-run/step-10` with another new
output directory and identical settings. A partial directory without a completed
training state is not resumable. The adapter alone is not a resume checkpoint.
The Python API accepts cooperative cancellation; phase-specific cancellation and
failure recovery tests remain outstanding.

Two real CPU/CUDA tests passed in 9.14 s: four steps uninterrupted versus two plus
two resumed match every tensor and final validation metric. Changed learning rate
is rejected before output creation, and parent bytes remain unchanged. A real
two-step CLI run exited zero in `artifacts/adapter-trainer-cli-v1`, with periodic
snapshots and a reserved-loss report. These fixtures establish mechanics, not
general quality or a measured resource advantage.

Interruption recovery was subsequently tested with real CPU updates. Cancellation
immediately after update one saved a resumable step-one checkpoint with absent
final evaluation. An injected exception before update two retained the previously
completed step-one checkpoint and produced no completion report. Resuming either
case to four steps matched every tensor from uninterrupted training. The four
trainer tests (including CPU/CUDA exact resume) passed in 13.01 s. This verifies
these boundaries, not arbitrary power loss, cancellation inside native kernels,
or failure during checkpoint publication.

A checkpoint-publication failure is now covered too: an injected exception after
writing step-two factors but before its training state leaves an incomplete
directory. Resume rejects it explicitly before creating a new output. Combining
that factor file with the valid step-one training state also fails because their
artifact hashes differ. Resuming the intact step-one directory still matches the
uninterrupted final tensors. All five trainer tests passed in 15.79 s. This tests
the two-file consistency boundary, not durability after an actual power failure.

## Measured resources

The trainer records synchronized update time and supervised (non-padding) tokens
per step. Report totals cover only newly executed steps after resume. Update
throughput excludes checkpoint writing and evaluation; a separate elapsed value
includes those phases after initial data preparation/baseline evaluation. RSS is
an end-of-run process sample, not a peak. CUDA values are allocator peaks reset
after baseline, and do not represent all driver or other-process GPU memory.

A twenty-step CUDA run with 448 trainable factors processed 1,190 supervised
tokens in 0.8623 s of updates (1,380 tokens/s). Training, saves and final evaluation
took 1.1597 s after setup; peak allocated CUDA memory was 17,529,344 bytes, reserved
23,068,672 bytes; final process RSS 1,450,520,576 bytes. Exact settings and report:
[ADAPTER_RESOURCES_CUDA.json](ADAPTER_RESOURCES_CUDA.json). Artifacts remain under
`artifacts/adapter-resources-cuda-v1`. This tiny single run is not a comparison with
full fine-tuning and establishes no speedup or community-node efficiency.
Five trainer tests passed in 18.17 s, including resumed step/token accounting and
CPU/CUDA metric availability, with exact weight recovery still passing.

## Installed-package reproduction

The wheel with SHA256
`9f09ad40e02368651971087731a61b43764db8e230dcb94e5c4092c98d03cf5e`
was installed in the isolated local environment. With Python `-I` and PyTorch
2.15.0+cpu, the forty-step adapter smoke completed, then separate CLI processes
trained two of four steps and resumed to four. All processes exited zero.
Artifacts: `artifacts/installed-adapter-smoke-v1`,
`artifacts/installed-adapter-partial-v1`, and
`artifacts/installed-adapter-resumed-v1`. The resumed report records only its two
new steps and 128 supervised tokens, with candidate visibility still restricted.
This installed run verifies process restart; exact uninterrupted/resumed tensor
comparison is established separately by the CPU/CUDA tests.

Foundation CI now includes these installed smoke and restart commands and retains
their reports. The workflow changes remain local and have not run remotely.

## Typed public gradient kernel

`foundation.adapter_tasks` adds a bounded adapter-gradient payload and executor,
reusing the existing public gradient base/data/runtime validation and FP32 tensor
codec. Both base weights and adapter factors require explicit public
redistribution declarations. Restricted local artifacts are not automatically
converted or submitted. Rank is limited to 16 and projection dimensions; only
q/v factor names/shapes are accepted. Factor checks precede base allocation.

The executor computes real token-normalized gradients for factors only. Its
artifact binds base weights, factors, tokenizer, settings and document hashes.
An independent CPU recomputation checks outputs using the same numerical
tolerances as the existing gradient protocol. Two CPU/CUDA tests passed,
including verification of actual gradients, changed token counts and rejection
of factors without public redistribution permission. This kernel is not yet
registered in API/scheduler/NodeAgent, does not aggregate updates or pay credits,
and does not demonstrate collaboration between machines. Full-base payloads and
server recomputation remain expensive reference-protocol limitations.

The local aggregation function now accepts two to eight payload/result pairs,
checks current base/factor/tokenizer/settings identity, recomputes each shard,
rejects overlapping documents and verifies that the optimizer owns exactly the
trainable factors. Only after every shard passes does it apply token-weighted
gradients, clipping and AdamW. A two-shard update matches the full-batch reference
within FP32 tolerance; duplicated shards and altered counts leave both model and
optimizer unchanged, and replay after an update is rejected as stale. Three
adapter-task tests passed in 9.18 s. API/scheduler/NodeAgent and credit integration
are still absent; this is local mathematical verification, not network execution.

The worker dispatcher now recognizes `foundation.adapter-gradient.v1` only with
the separate `allow_foundation_adapters=True` argument. Existing training or
evaluation permission does not imply this new permission; integer `1` is also
rejected. NodeAgent retains the option (default false) and passes it to execution,
but does not yet advertise the capability. The scheduler/API cannot create these
jobs yet. This staged integration therefore does not enable new remote work on
existing nodes. Tests exercise real dispatch and the independent verifier, along
with existing node and tokenization behavior.

Capability negotiation now recognizes the adapter type and requires a PyTorch
runtime declaration. NodeAgent advertises it only with its separate opt-in;
default nodes keep their previous capabilities. Server capability updates retain
that runtime. Scheduler dispatch explicitly excludes adapter jobs until creation,
verification and server-side enablement are integrated. This deliberate dispatch
gate prevents capability registration alone from enabling work. The HTTP-client
negotiation test validates the actual JSON against NodeCapabilities; existing
node/job/distributed regression tests also pass (36 tests in 18.75 s). End-to-end
adapter job submission and execution are still pending.

## API and scheduler integration

`POST /api/v1/jobs/foundation/adapter-gradient` now creates validated public
adapter jobs under `foundation_distributed_adapters_enabled` (default false).
The request uses the same fields as evaluation jobs: payload, idempotency key,
explicit public-node consent and optional reward credits. Creation/reservation,
leases, independent result verification and settlement reuse the existing job
transaction path. Capability, available RAM/VRAM and nested base PyTorch version
filter assignment. Worker adapter opt-in remains separate.

Six adapter-task tests passed in 10.20 s. The FastAPI in-process integration test
checks disabled creation, consent rejection, idempotency, runtime-incompatible
node exclusion, real worker gradients, invalid-result rejection and verified
completion. That run uses zero reward; funded credit behavior and two-process
TCP adapter execution have not yet been demonstrated. This is not a multihost
claim. Existing performance scheduling profiles are not reused for adapter jobs.

Funded behavior was then verified in-process with separate owner and volunteer
accounts. From 100 test credits, creation reserves nine once; an invalid gradient
earns nothing, verified completion pays nine, and resubmission cannot pay again.
A second job's cancellation refunds its reservation once and rejects cancellation
by the volunteer. Insufficient funds reject creation, and total ledger balance
remains conserved. Sixteen adapter/credit tests passed in 14.67 s. These are
SQLite test transactions with synthetic credits, not money, and do not establish
PostgreSQL concurrency or multi-process TCP execution for adapters.

## Real HTTP worker experiment

The subsequent TCP experiment completed two jobs through the actual API with
separate CPU and CUDA worker processes on one Windows machine. Independent
verification accepted 144 supervised tokens (mean loss 5.652894761827257), and
aggregation changed the factors by L2 norm 0.006795629506006286. Each volunteer
received three synthetic credits; the owner's six-credit balance and reservations
were exhausted exactly. The restricted adapter and merged candidate were saved;
neither was promoted. See [recorded report](ADAPTER_DISTRIBUTED_SMOKE.json).

Reproduce from source with `PYTHONPATH=services/api/src` and:

```text
python -m ai_home.foundation.adapter_distributed_smoke --output-dir artifacts/adapter-distributed-new --worker-devices cpu cuda
```

Use a new output directory. CPU-only systems can use `--worker-devices cpu cpu`.
Workers execute sequentially; this verifies HTTP transport, process separation,
gradient aggregation and credits, not multihost operation or parallel speedup.
The installed-package CPU variant is now configured in CI, but this local change
has not yet run in remote CI. Worker CLI/desktop adapter opt-in remains pending.

The installed-package CPU variant subsequently passed locally with Python 3.11.9
in isolated mode (`-I`) and PyTorch 2.15.0+cpu. An explicit import check resolved
`ai_home` inside the reproduction environment's `site-packages`. The built wheel
was 270542 bytes, SHA-256
`ad3a27aad212019b940d28f00316f2dac86bc6a16ae4e7b89a2d6a59a9823121`.
Two separate CPU workers completed the HTTP jobs, verified 144 tokens and earned
three credits each. Factor delta was 0.00679562816765376; the merged candidate
remained unpromoted. See [installed report](ADAPTER_INSTALLED_DISTRIBUTED_SMOKE.json).

```text
python -m pip install . --no-deps --force-reinstall
python -I -m ai_home.foundation.adapter_distributed_smoke --output-dir artifacts/installed-adapter-distributed-new --worker-devices cpu cpu
```

Run these commands in an environment with project dependencies already installed.
The source CPU/CUDA experiment used PyTorch 2.14.1+cu130; this second run adds
installed CPU coverage, not cross-version worker compatibility within one job.
The optional missing-NumPy warning did not prevent completion. Remote CI remains
pending, and both experiments are confined to one physical host.

## Desktop worker consent

DISTRIBUTED COMPUTE now offers a separate adapter-training checkbox, disabled
by default. Starting the node captures that permission and the chosen CPU/CUDA
device; changing permissions requires stopping and restarting the node. Switching
accounts stops the node and clears the adapter permission along with existing
permissions. Training/evaluation consent does not enable adapters.

The real Tk worker experiment completed an adapter job on CUDA over loopback HTTP,
with server verification and three synthetic credits. It checked default-disabled
consent, exclusive capability advertisement, profile reset and cross-account node
reuse rejection. See [desktop report](ADAPTER_DESKTOP_CUDA.json). Reproduce with
`PYTHONPATH=services/api/src`:

```text
python scripts/verify_desktop_node.py --output-dir artifacts/desktop-adapters-new --device cuda --task adapter-gradient
```

The real Tk navigation verifier also invoked the adapter checkbox and confirmed
its independence from the other permissions. These hidden-window checks prove
behavior, not visual layout acceptance. The installed CPU desktop variant is
configured in CI; this new desktop change has not yet run remotely.

Malformed-result regression: the adapter verifier now rejects excessively nested
artifact JSON without propagating `RecursionError`, checks the artifact shape,
requires integer token counts and finite numeric loss before neural recomputation.
The funded API test submits nested JSON, null, arrays and truncated JSON: each
returns HTTP 422, leaves the worker unpaid and preserves the owner's reservation;
a subsequent valid result still settles exactly once. Sixteen adapter/credit
tests passed in 11.37 s, then the expanded eight-test adapter suite passed in
7.35 s, including rejection before neural execution for wrong types/nonfinite loss.
This covers these malformed-result cases, not an exhaustive protocol audit.

## Evaluation of the distributed candidate

The saved parent and merged distributed candidate were evaluated using the ten
existing authored general probes, greedy generation, 64 new tokens and CUDA.
Both scored 0/10. Their 16-token context truncates prompts, so this experiment
cannot establish general capability or non-regression. Full outputs are preserved
in `ADAPTER_PARENT_GENERAL_EVALUATION.json` and
`ADAPTER_CANDIDATE_GENERAL_EVALUATION.json`.

A separate CPU comparison reloads both checkpoints, verifies that the two held-out
documents are in validation lineage and absent from training hashes, and scores
72 supervised tokens. Parent loss was 5.649996439615886; candidate loss was
5.649986902872722. This tiny numerical change on near-identical authored fixture
templates is not evidence of language improvement or out-of-distribution
generalization. Neither checkpoint is promoted. Report:
`ADAPTER_DISTRIBUTED_HOLDOUT.json`. Reproduce with source on `PYTHONPATH`:

```text
python scripts/evaluate_adapter_distributed.py --experiment artifacts/adapter-distributed-v1 --output artifacts/adapter-distributed-v1/holdout-new.json
```

The distributed experiment now accepts `--context` (integer 8–128, default 16).
A subsequent `--context 64 --worker-devices cpu cuda` run completed both real
HTTP jobs and payouts. Its freshly trained parent and merged candidate both
scored **0/10 with all prompts fitting** under the same unchanged general suite,
greedy decoding and 64-token generation budget. Thus removing the truncation
limitation did not demonstrate improved answers. Equality at zero is not proof
of preserved general ability. CPU holdout loss changed from 5.648322211371528 to
5.648314158121745 on the same 72 synthetic tokens; no promotion followed.

Evidence: `ADAPTER_CONTEXT64_DISTRIBUTED.json`,
`ADAPTER_CONTEXT64_PARENT_EVALUATION.json`,
`ADAPTER_CONTEXT64_CANDIDATE_EVALUATION.json`, `ADAPTER_CONTEXT64_HOLDOUT.json`.
This changes context during initialization and training; it does not mutate an
existing checkpoint's configuration. The experiment remains one host, sequential
workers and a small synthetic corpus, with no claim of general learning.

## Desktop report inspection

The existing EVALUATION "Abrir informe" viewer accepts local adapter-training
reports. It shows completed/target steps, trainable parameter count and reserved
loss before/after (or cancellation without final evaluation). The read-only JSON
includes resource measurements and provenance. Reports remain untrusted local
content; inspection neither verifies execution nor activates a checkpoint.

The bounded reader validates lifecycle, parent digest, checkpoint name, numeric
metrics, matching holdout token counts and available resource fields. Fourteen
report tests passed, including altered lifecycle and invalid resource values.
The real Tk test loaded the recorded CUDA report exactly, checked the visible
summary and read-only content, and closed it on profile switch. Reproduce:

```text
python scripts/verify_preference_report_desktop.py --report artifacts/adapter-resources-cuda-v1/report.json
```

Despite its historical filename, this verifier supports preference and adapter
reports. Installed-package adapter inspection is configured in CI but has not
yet run remotely. This hidden Tk check does not establish visual layout quality.

## Local training form

TRAINING now includes "Entrenar adaptadores": parent checkpoint, authorized
public dataset, new output directory, optional step-N resume directory, CPU/CUDA
and trainer settings. Work runs in the existing background mechanism; cancellation
is cooperative between updates, and closing the private window requests it.
The form shows terminal status and the full report; live per-step progress is
not yet exposed. It never merges, uploads or promotes weights automatically.

The real Tk verifier trained two updates, cancelled a second run after one real
update, resumed its saved checkpoint and compared every final factor exactly
with the uninterrupted run. It also checked closure on profile switch after
training. This does not yet test profile switching during an active adapter run.

```text
python scripts/verify_adapter_training_desktop.py --fixture-dir artifacts/adapter-lineage-cpu-v1 --output-dir artifacts/adapter-desktop-training-new
```

The source CPU run passed. Installed CPU execution is configured in CI but has
not yet run remotely; visual layout acceptance remains pending.

The form subsequently gained live step/loss/tokens-per-second updates from actual
completed optimizer steps. The trainer emits a copy of each metric; a one-item
queue forwards only the latest measurement to Tk's main thread. Completion stops
polling, and changing session or destroying the window requests cancellation.

The expanded real Tk verifier paused the second actual optimizer update and
observed step 1/2 plus tokens/s before allowing completion. It then repeated
cancellation and exact resume, and switched profile during another active update.
That run terminated at step one with a saved restricted checkpoint and no final
evaluation; its queued callback was discarded without errors after the window
closed. Source verification passed with artifacts in
`artifacts/adapter-desktop-progress-v1`. This supersedes the earlier limitations
on live metrics and active-profile behavioral verification, but not visual review
or remote CI execution.

## Installed desktop reproduction

The combined desktop changes were rebuilt into an installed wheel (273772 bytes,
SHA-256 `a0ac3ca8dd235f77f2bf734191548b104eb40bc06eaa1dee156fa72d99332d22`).
Python 3.11.9 with `-I`, explicit site-packages assertions and PyTorch 2.15.0+cpu
passed the training verifier including live metrics, cancellation, exact resume
and active profile switch. The report verifier then loaded that resumed run;
the desktop node separately completed a verified CPU adapter job and earned
three synthetic credits. Outputs are under `artifacts/installed-adapter-ui-v1`
and `artifacts/installed-desktop-adapters-v1`. These are local installed-package
results; remote CI and visual layout review remain pending.

PostgreSQL adapter settlement is now covered by a targeted concurrent test:
two simultaneous valid submissions of one leased result produce one payment of
nine synthetic credits and one already-completed response. The eight-test local
PostgreSQL suite passed in 5.15 s, and the dedicated portable cluster was stopped
afterward. See [PostgreSQL evidence](POSTGRES_VALIDATION.md). This supersedes the
earlier absence of PostgreSQL adapter testing, while broader concurrency and
multihost behavior remain unverified.

## Export for local inference

The factor-to-checkpoint export is now available as an explicit command. It
requires the matching parent and validated lineage, preserves the restricted
candidate status and refuses to overwrite an existing output:

```text
python -m ai_home.foundation.adapters export --parent artifacts/adapter-lineage-cpu-v1/parent.pt --adapter artifacts/adapter-desktop-progress-v1/resumed/step-2/adapter.pt --output artifacts/adapter-desktop-progress-v1/exported-candidate.pt
```

This command was executed on the candidate trained/resumed through Tk. The
resulting ordinary checkpoint was loaded by the installed API with isolated
Python. Real loopback requests verified authentication rejection (401), a
completion and SSE streaming. Both generated incoherent replacement characters;
this demonstrates inference integration only. No registry approval or global
activation occurred. Recorded API output: `ADAPTER_INSTALLED_API.json`.
Exported checkpoint SHA-256:
`707e1c8db0d4344ed5a7c8f721d2c7ab003be1a1126183e1de2ed3a2a04b91ac`.
Six adapter serialization/merge tests also passed in 2.94 s.

```text
python -I scripts/verify_installed_api.py --output-dir artifacts/installed-adapter-inference-new --checkpoint artifacts/adapter-desktop-progress-v1/exported-candidate.pt
```

Run the second command from an environment with the application installed.
The local adapter source command is new; the installed inference path already
supports the ordinary checkpoint format and was tested without rebuilding it.

## Response-only supervised adapter training

`adapter_training --objective sft` reuses the Foundation SFT sequence builder:
user text and role markers are masked, while assistant response tokens and EOS
contribute to the loss. Every input record must contain validated conversation
messages whose content matches its reviewed text. Existing public authorization
and privacy checks remain mandatory; this does not admit private review exports.

SFT is recorded in resume identity and the report. Switching objective on resume
is rejected before creating an output directory. The absent objective in earlier
training checkpoints continues to mean causal training for compatibility.
Seven trainer tests passed in 7.90 s, including CPU/CUDA response-token accounting
(two A/EOS responses: four supervised tokens per update), exact resume and
objective-mismatch rejection. These synthetic examples verify training behavior,
not acquired instruction-following ability. The desktop form currently retains
the default causal objective; SFT selection is available through the trainer API
and CLI.

The desktop form now also offers explicit causal/SFT selection, explains which
tokens each objective supervises, and captures the choice before background work.
The report reader rejects an objective that disagrees with resume identity; old
reports without this field retain causal semantics. The viewer labels the objective.

A real Tk SFT run on eight authored public conversation fixtures passed live
progress, cancellation after an update, exact resumed factors and active-profile
isolation. The resumed step counted four supervised tokens (two A/EOS responses).
Its report loaded through the desktop viewer. Artifacts are preserved under
`artifacts/adapter-desktop-sft-v1`; reproduce with the existing verifier plus
`--objective sft`. The installed CPU variant is configured in CI but not yet run
remotely. No improvement in general instruction following is claimed.

## Concurrent worker launch

The HTTP experiment now supports `--parallel`, launching both subprocesses before
waiting for either result. A `--context 64 --worker-devices cpu cuda --parallel`
run completed both jobs, independently verified 144 tokens, paid three credits to
each worker and changed the factors (L2 0.006812654773524994). All saved factors
and data lineage matched the equivalent sequential run exactly. Evidence:
`ADAPTER_CONCURRENT_WORKERS.json`.

Worker subprocesses are collected on success and killed/reaped during cleanup
after failures or timeouts. The default remains sequential; the installed CPU
CI experiment now selects concurrent launch. Launching concurrently does not
prove kernel overlap or speedup, and this remains a one-host loopback experiment,
not multihost validation. No model was promoted.
