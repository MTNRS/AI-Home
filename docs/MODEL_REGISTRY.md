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

The registry has no public mutation API or remote publication. Category-by-category
regression comparison, approved-suite policies, signatures, revocation, desktop
version selection, and multi-host deployment remain outstanding. Stored content
hashes detect changes, not malicious modification by someone with registry write
access. The full checkpoint is hashed on each selection, which favors verification
over latency; production-scale serving will require immutable deployment storage
and measured caching. Current registration file budget defaults to 2 GB.
