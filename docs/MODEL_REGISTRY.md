# Foundation model registry

The local operator registry separates candidate creation, evaluation, approval and
activation. It copies checkpoints into content-addressed storage and evaluates
those retained bytes with the existing Foundation Evaluation Engine. It retains
the full probe suite, exact checkpoint/tokenizer hashes, generation settings,
runtime metadata and individual results. No candidate-supplied report is trusted
as the evidence for registration.

```powershell
$env:PYTHONPATH = "services/api/src"
python -m ai_home.foundation.registry --registry artifacts/models register candidate.pt docs/evaluation-probes-v1.json --device cuda
python -m ai_home.foundation.registry --registry artifacts/models list
python -m ai_home.foundation.registry --registry artifacts/models inspect VERSION_ID
python -m ai_home.foundation.registry --registry artifacts/models approve VERSION_ID --operator reviewer-id
python -m ai_home.foundation.registry --registry artifacts/models activate VERSION_ID --operator operator-id --reason "Reviewed release"
python -m ai_home.foundation.registry --registry artifacts/models history
```

Approval requires an operator identity, known training-document lineage, every
probe prompt fitting the context and the selected accuracy threshold (default
1.0). The threshold is explicitly configurable and recorded. A repeated approval
cannot overwrite the original reviewer or threshold. Passing a small suite is
not a general model-quality certification. Suite quality and adequate independent
human review remain operator responsibilities. These identities are local audit
labels, not cryptographic authentication or a multi-user approval service.

Activation requires an approved candidate and records an append-only event with
operator, reason and timestamp. Rollback uses the same activation command with a
previous approved version ID. Neither action mutates weights. SQLite transactions
serialize activation and approval writes. Checkpoint publication uses a temporary
file and an exclusive hard link on the registry filesystem; interrupted temporary
copies cannot replace an existing content-addressed checkpoint. A failed evaluation
may leave unreferenced weights; automated garbage collection is not implemented.

## Server integration

Configure `AI_HOME_FOUNDATION_REGISTRY` instead of
`AI_HOME_FOUNDATION_CHECKPOINT`. Each Foundation request selects and verifies the
latest approved activation. A request already running keeps its selected version;
subsequent requests observe later activation/rollback. Missing, unapproved,
corrupt or ambiguous configuration returns the existing generic inference error;
there is no external-model fallback. Direct checkpoint mode remains supported
for development and does not imply registry approval. Configuring both modes is
an error. Read access to the registry database and weights is assumed trusted;
protect filesystem permissions and backups accordingly.

## Validation and limits

Tests train two real tiny checkpoints, register/evaluate them, verify that the
default quality gate rejects them, exercise explicit zero-threshold lifecycle
approval, activate both, roll back, run real authenticated API generation, corrupt
the retained bytes, and verify fail-closed inference without path disclosure.
Zero-threshold approval is test-only evidence of lifecycle mechanics. No useful
conversational capability or production release is claimed.

The registry has no public mutation API or remote publication. Approved-suite policies, signatures, revocation and multi-host deployment remain outstanding. Stored content
hashes detect changes, not malicious modification by someone with registry write
access. The full checkpoint is hashed on each selection, which favors verification
over latency; production-scale serving will require immutable deployment storage
and measured caching. Current registration file budget defaults to 2 GB.


## Regression gate for subsequent approvals

The first approved version is recorded as a bootstrap approval. Subsequent
approvals compare against the active approved version, or the most recently
approved version when none has been activated. An operator can explicitly select
another approved baseline with `approve ... --baseline VERSION_ID`; that choice
is retained in the approval record and is not an automatic claim of comparison
to the active model.

The reports must match suite hash, generation settings, device and PyTorch version,
with all prompts fitting. The gate rejects every previously passing probe that
now fails, even if the total score or category total stays unchanged. The retained
comparison includes category deltas, gained/lost probe IDs and both checkpoint
hashes. Approval plus comparison are committed in one SQLite transaction; a failed
gate leaves the candidate unapproved. `inspect` includes this comparison.

Existing approved versions created before this extension remain approved and have
no retroactively invented comparison. Rollback can still reactivate a previously
approved version without pretending it is a new improvement. This gate covers the
fixed declarative probes only; it does not measure catastrophic forgetting across
all knowledge or replace held-out loss/regression experiments. Changes to the
suite/runtime require a deliberate evaluation policy transition; comparisons are
not silently accepted across incomparable runs.


## Authenticated model selection

`GET /api/v1/models` returns only approved registry versions with their UUID,
checkpoint hash and active flag. It requires an authenticated account and exposes
no filesystem paths, reviewer identities, evaluation prompts or generated text.
Without a Foundation registry it returns an empty catalog. Catalog entries are
approval records; checkpoint integrity is verified when selected for inference.

`POST /api/v1/chat/completions` accepts optional `model_version` (UUID). Selecting
an approved version affects only that request, not global activation. Without an
explicit version the active version is resolved once for the request. The response
includes the actual `model_version`; direct checkpoint and external-provider modes
return null. Unknown/unapproved/corrupt selections fail without falling back to
another model. Explicit version selection requires registry-backed Foundation
serving. Clients can pin this returned UUID to keep a conversation on one version.

Validation includes real generation from an explicitly selected non-active
checkpoint, unchanged global activation, catalog authentication, no candidate
leakage, and rejection of unapproved selection. Desktop selection is available as described below.


## Desktop selection and conversation pinning

In provider routing, “Actualizar versiones aprobadas” fetches the authenticated
server catalog. The read-only selector offers the active-server default and
approved version UUIDs. This catalog and selection belong to the current account
session and reset when the profile changes.

The first registry-backed response pins its returned UUID for the conversation.
Later requests send that UUID even if the server activates another version.
Selecting a different version requires clearing the conversation. An explicitly
selected version must match the response version; the client rejects mismatches
or missing version metadata. Servers/direct backends that return no version
remain supported but cannot provide registry pinning guarantees.

The conversation source records `AI@Home:VERSION_UUID` for subsequent consented
contribution provenance. Clearing chat resets the pin and selector. The selector
does not activate or approve any version globally. Tests verify catalog auth,
response mismatch rejection, second-turn pinning and provenance; a hidden real
Windows Tk check verifies widget construction, chat reset and profile reset.
Full suite: 194 passed, 4 PostgreSQL skipped. Visual layout review remains pending.
