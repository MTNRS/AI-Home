# Experimental preference objective

The own Foundation model now has a low-level response log-probability and DPO
loss implementation in `foundation/preference_objective.py`. It follows the
reference-relative pairwise objective from
[Rafailov et al., Direct Preference Optimization](https://arxiv.org/abs/2305.18290).
No third-party weights, tokenizer or reward model are introduced.

`response_log_probability` sums only the explicitly supplied response tokens.
Prompt targets are excluded; callers must include EOS in the response. There is
no silent truncation, padding or length normalization. `dpo_loss` detaches both
reference scores and rejects incompatible shapes and non-finite inputs/losses.
These functions do not grant permission to use any input data.

Three tests passed locally, including real randomly initialized model updates on
CPU and CUDA. Twelve AdamW steps increased the preferred-versus-rejected response
log odds on one authored toy pair; reference weights remained unchanged and
received no gradients. Separate checks cover prompt masking, objective-gradient
direction, shape validation and context overflow. This establishes numerical
mechanics, not human alignment, conversational improvement or generalization.

The existing consented correction export does not record an independently reviewed
preferred/rejected pair. It is not automatically converted into preference data.
Independent language evaluation and operator-facing integration remain required
before this becomes a complete training workflow. Training is not exposed through API
or volunteer jobs, and
does not promote or alter deployed models.

## Reviewed-pair import

Prepare an unapproved review sidecar from an already consented, reviewed export:

```powershell
python -m ai_home.foundation.preference_data --export frozen.json --output pending-pair-review.json
```

The command validates the export, groups identical normalized prompt histories,
deduplicates identical responses and lists eligible pairs in deterministic ID
order. That order is **not** a quality preference. Review both responses in the
source export, reorder chosen/rejected IDs as appropriate, remove unwanted pairs,
and enter the actual reviewer identity and review epoch before confirming each
retained pair. The template deliberately leaves identity empty, epoch zero and
all confirmations false, so training rejects it unchanged. These fields record
an operator assertion; they are not authenticated proof of human review.

The command refuses existing output files and more than 1000 eligible pairs.
It does not copy response text into the sidecar or send anything over the network;
the IDs and manifest still belong to restricted data and must be kept private.
Eleven data tests passed. A real CLI invocation on the authored smoke export
created twelve unapproved pairs at `artifacts/preference-review-template-v1.json`.

The desktop training form exposes the same command as `Preparar revisión`.
Choose the consented export and a new sidecar file; the form reports the number
of pending pairs and leaves the training-review selection unchanged. Complete
the review separately before selecting it for training. Template preparation and
training buttons are mutually disabled while either operation runs. Closing the
window discards its callback but may leave a completed template on disk; the
training Cancel button does not interrupt this short file-preparation operation.
The real Tk verifier created twelve unapproved pairs, verified that the importer
rejected them, then completed training, live progress, cancellation, exact resume
and profile isolation checks in `artifacts/desktop-preference-template-v1`.

`foundation.preference_data.load_preference_pairs(export_path, review_path)` reads
the frozen export once (8 MB limit) and reuses the existing consent, rights,
privacy, integrity and ancestry validation. The separate review is limited to
1 MB/1,000 pairs, has format `ai-home-preference-review-v1`, and requires
`export_manifest_sha256`, `reviewer_id`, `reviewed_at_epoch` and `pairs`. Each pair
names `chosen_contribution_id`, `rejected_contribution_id` and an explicitly true
`preference_confirmed`. Both IDs must exist in that validated export and pair
review must not predate either source review.

Only identical normalized prompt histories with distinct nonempty final assistant
responses are accepted. Duplicate or opposite orderings of the same pair are
rejected. The loader retains both contribution IDs, source-family groups, a stable
prompt-family hash, export/review hashes, and `reviewed-training-only` visibility.
Nothing is sent to volunteer nodes or made public. Hashes bind content, not the
identity or authority of a reviewer; operators must use authenticated exports and
trusted review files. There is no API to create/review these pairs yet.

Integration checks exercise restricted output, stable prompt grouping, contradictory
pairs, wrong prompts, missing consent, unknown IDs and unbound/early reviews. Existing
frozen export and correction tests are included to verify the shared validation
refactor. No actual private conversation was used in these fixtures.

## Family partitions

`split_preference_pairs` joins pairs sharing either a prompt-family hash or any
source-conversation family, including transitive connections. It assigns each
whole connected component by a deterministic hash of its sorted family IDs and
returns canonically ordered training/validation rows. Input order cannot change
the result; inputs are not mutated. Empty partitions fail explicitly rather than
splitting a family to meet a ratio. Counts need not match the requested fraction.

Tests cover a chain joined partly by source ancestry and partly by prompt, plus
independent families, permutation invariance, complete row coverage and zero
family overlap. Adding new links can merge components and change assignments;
freeze and retain the resulting partitions for comparisons. This does not discover
semantic similarity or compensate for missing/wrong provenance. The splitter is
for already validated pair records; it is not a substitute for the import checks.

## Local candidate training

`foundation.preference_training.train_preferences(parent, export, review, output_dir)`
loads a bounded own checkpoint, verifies document/group lineage, imports reviewed
pairs, and splits connected families before any updates. It rejects holdout families
already trained by the parent and new training families reserved for parent
validation. CPU and explicit CUDA are supported with a device-memory estimate
covering both policy and frozen reference. The tokenizer is unchanged. Prompts and
both responses must fit the model context; no truncation is permitted.

The bounded reference implementation uses FP32 AdamW, gradient clipping and finite
loss/weight checks. It samples one pair per update with a private seeded generator.
Outputs go to a new directory: a normal loadable Foundation `candidate.pt` and a
pair-likelihood report. Both remain `reviewed-training-only`; no private text is
copied into checkpoint/report metadata, no volunteer dispatch or promotion occurs.
The original parent file is not changed. Reports include the candidate checksum,
input/configuration identity, fixed-reference metrics and candidate holdout metrics.

`stop_after` saves an intermediate checkpoint; `resume` requires identical parent,
reviewed data, partition, settings, device and PyTorch version, then restores AdamW
and sampler state. `checkpoint_every` (default 10) saves complete `step-N.pt`
snapshots with optimizer, sampler and input identity; final/controlled-stop runs
also save `candidate.pt`. Cancellation is checked between pairs;
after an update it saves a restricted candidate and skips final evaluation. Active
kernels cannot be interrupted. Cancellation during final evaluation retains the
already-saved checkpoint and records absent candidate metrics.

Tests perform real CPU/CUDA updates, compare uninterrupted and resumed weights
exactly, check unchanged parent bytes and increasing held-out preference margin
on authored toy pairs, and reject changed resume settings and parent contamination.
A controlled cancellation after a real optimizer update saves step one. These are
mechanical demonstrations on fixtures, not evidence of human preference quality,
language improvement, robustness to poisoning or privacy-preserving unlearning.

## Command line and interruption recovery

```powershell
python -m ai_home.foundation.preference_training --parent parent.pt --export frozen.json --review pairs-review.json --output-dir candidate-run --device cpu --steps 20 --checkpoint-every 5
```

Use `--stop-after` for a controlled partial run. Resume with `--resume
candidate-run/step-10.pt` (or `candidate.pt`) and a new output directory, retaining
the same settings including checkpoint interval. Other controls are
`--learning-rate`, `--beta`, `--seed` and `--validation-fraction`. Ctrl-C exits 130;
only checkpoints whose save completed before interruption are available. It does
not claim to preserve an in-flight update.

Seven trainer tests passed locally, including the real CLI in separate processes,
CPU/CUDA resume, cancellation and contamination rejection. A controlled exception
before update three left a complete step-two snapshot; resuming it produced
exactly the same final CPU weights as uninterrupted training. This verifies that
recovery boundary, not all possible filesystem/power-failure scenarios.

Resume also checks the loaded policy architecture against the frozen reference,
not just its declared experiment identity and tokenizer. A modified context size
can retain valid tensor shapes while changing model behavior; the trainer rejects
that case before creating an output directory. Resume loads its policy directly
without first allocating an unused copy of the reference model.

## Preserved synthetic run

`python -m ai_home.foundation.preference_smoke --output-dir NEW_DIR --device cuda`
creates authored A-versus-B fixtures, a small own-model parent, explicit synthetic
review records, and trains eight pairs with four reserved prompt families. All
synthetic review identities say `SYNTHETIC-FIXTURE-NOT-A-HUMAN-REVIEW`; no real
conversation or human preference is represented. The run stops at step ten and
resumes to twenty, saving periodic checkpoints and an unscored generation.

Both CPU and CUDA runs completed locally. On the four reserved pairs, the fraction
with higher likelihood for A than B rose from 0/4 to 4/4; mean log-probability
margin changed from -0.043695 to about 5.390419. Greedy generation for a new numeric
prompt produced A. This trivial shared-label task demonstrates the optimization
mechanism, not semantic generalization, language quality or alignment with people.
Full reports: `PREFERENCE_SMOKE_CPU.json` and `PREFERENCE_SMOKE_CUDA.json`.

The pipeline still labels these candidates `reviewed-training-only`, with no
automatic promotion or volunteer distribution. Artifacts, including synthetic
export/review files and checkpoints, remain under `artifacts/preference-smoke-*`.
The CPU smoke has been added to installed-package CI, but these uncommitted changes
have not been remotely validated.

The installed package was also exercised with isolated Python (`-I`) in
`artifacts/reproduction-venv-v1`, using PyTorch `2.15.0+cpu`. Imports resolved to
`site-packages`; `pip check` reported no broken requirements. The standalone
smoke completed twenty steps including the ten-step restart, with four held-out
pairs improving from 0/4 to 4/4 and mean margin 5.390418767929077. Its report and
checkpoints are preserved in `artifacts/installed-preferences-v1`; candidate SHA256
is `f6866db4c63ded59f2807a98ef8e64041062704a191b5cb8ca6edb31820dc8a6`.
This is a local installed-package reproduction, not a clean-machine or remote CI
result. PyTorch emitted its optional missing-NumPy warning; the run exited zero.

## Desktop report inspection

TRAINING now offers `Abrir resultado de preferencias` using the existing bounded
local report viewer. Select the trainer's `report.json` (not the outer smoke
summary). It displays steps and baseline/candidate held-out margins, or the
absence of final evaluation after cancellation. The reader checks hashes,
restricted visibility, no promotion, step consistency, finite metric ranges and
matching held-out pair counts. Completing all optimizer steps can coexist with
cancelled final evaluation. Inspection does not verify that training occurred or
that the checkpoint matches the report; the existing provenance notice remains.

All thirteen report-reader tests passed, including preserved real DPO results,
cancelled final evaluation and six inconsistent-report cases. This new desktop
entry has not yet received a visual UI check. It does not launch DPO training.

`scripts/verify_preference_report_desktop.py` now exercises the actual Tk button,
background reader, exact loaded JSON, summary text, read-only content and profile
switch cleanup. It passed against the installed CPU experiment's trainer report
using current source code. The first run exposed loss of the summary StringVar
after the asynchronous callback returned; retaining it on the report window fixed
the issue. This checks widget behavior with hidden windows, not visual layout.
The same check is configured for installed-package CI; that CI has not run yet.

## Desktop training

TRAINING offers `Entrenar preferencias`: select the parent checkpoint, frozen
consented export, explicit pair review and a new output directory. CPU/CUDA,
steps, learning rate, beta, seed, validation fraction, checkpoint interval and
optional resume use the same trainer as the CLI. Step and DPO loss update live;
the full report appears at completion. Updates pass through a one-item queue so
the worker never accesses Tk and display lag cannot accumulate unbounded events.
The trainer passes a copy of each metric after its update and periodic save;
display consumers cannot mutate checkpoint history. These are training losses,
not held-out evaluation results.
closing the window or changing profile requests cooperative cancellation.
Active kernels are not interrupted. No review is manufactured by the form.

`scripts/verify_preference_training_desktop.py` ran two actual CPU updates through
the Tk button using authored smoke inputs. It verified the saved restricted
candidate, report, restored button, completion status and profile-switch closure
in `artifacts/desktop-preference-training-v1`.

The extended verifier subsequently exercised the actual Cancel button after a
real AdamW update, checked the saved step-one candidate and absent final metrics,
then resumed through the same form. Every final tensor exactly matched the
uninterrupted two-step run. Evidence is in
`artifacts/desktop-preference-resume-v1/{full,cancelled,resumed}`. Synchronization
in the verifier pauses after the update solely to make button timing deterministic;
it does not replace training or generate synthetic weights. The process exited
zero. Installed-package CI now includes this verifier but has not run remotely.
These hidden-window checks still do not verify visual layout or cancellation
during every phase of training.

The verifier also switches profiles during a paused real optimizer update and
waits for the background operation's terminal event. It checks step-one
cancellation, restricted artifacts, absent final metrics, destroyed private
windows and discarded late callbacks. Both source and isolated installed-package
runs passed (`artifacts/desktop-preference-profile-v1` and
`artifacts/installed-preference-profile-v1`). The installed report-viewer check
also passed against the resumed candidate. Both installed processes exited zero;
the rebuilt wheel SHA256 was
`fe92e4c538d2b8dcf61ed99e9eda3fb5b9af780646b1645a6fdbc3e4cb97b8be`.
This proves local UI lifecycle behavior, not filesystem isolation between OS
users: operator-selected files remain on disk after profile changes.

Live progress was subsequently tested with real updates: the verifier pauses the
second optimizer update and checks that Tk displays the completed first step and
its DPO loss before allowing completion. It then repeats cancellation, exact
resume and active profile-switch checks. The source run exited zero at
`artifacts/desktop-preference-progress-v1`; seven trainer tests passed in 19.85 s,
including CPU/CUDA metric-history equality and protection against callback
mutation.

The latest installed-package verifier also passed with live progress and review
template creation, using isolated Python and PyTorch 2.15.0+cpu. Artifacts:
`artifacts/installed-preference-template-v1`; rebuilt wheel SHA256:
`48b952f95314ca021770569b41d05dd249187d0c869e340b717f63ca9ca02378`.
Installation and verification exited zero. The optional missing-NumPy warning
did not prevent execution.

A native screenshot review was attempted, but the desktop capture tool returned
another application's pixels for the selected Tk window, including after target
reselection. No visual acceptance is claimed. The preview process was stopped
and its termination verified; functional hidden-window checks remain valid.

## Independent generation result

The installed CPU smoke parent and twenty-step DPO candidate were evaluated with
the existing `evaluation-probes-v1.json`, greedy decoding and 64 generated tokens
per case. Both scored **0/10**, and all prompts fit the context. Reports are
[parent](PREFERENCE_PARENT_GENERAL_EVALUATION.json) and
[candidate](PREFERENCE_CANDIDATE_GENERAL_EVALUATION.json). Both CLI processes
exited zero with isolated Python and PyTorch 2.15.0+cpu. Report schema, loaded
checkpoint hashes and identical suite fingerprints were checked separately.

The synthetic A/B preference gain therefore has no measured benefit on these
independent language/instruction probes. Equal zero scores do not establish
general non-regression: the suite is small and already at its scoring floor.
The candidate remains restricted and unpromoted. Representative authorized
language training and meaningful preference data are still required.
