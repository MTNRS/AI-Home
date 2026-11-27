# Experimental Nano distribution

`ai_home.foundation.smoke_package` builds an immutable ZIP of the real mechanical
smoke experiment. It includes the checkpoint (weights, optimizer and resume state),
standalone own BPE tokenizer, architecture/training configuration, authored public
Spanish corpus, metrics/generated sample, Apache-2.0 license, reproduction notes,
and a manifest with file sizes and SHA256 identities. It is a candidate, not an
approved model release or a language capability demonstration.

From a source checkout, after installing `.[llm]` with an appropriate PyTorch build:

```powershell
python -m ai_home.foundation.smoke --output-dir artifacts/package-run --device cpu
$revision = git rev-parse HEAD
python -m ai_home.foundation.smoke_package build artifacts/package-run artifacts/nano-smoke.zip --revision $revision
python -m ai_home.foundation.smoke_package verify artifacts/nano-smoke.zip
```

The source revision must be a full commit SHA; it is supplied provenance, not an
attestation that the worktree was clean. Use a clean checkout of that revision for
reproduction. CI supplies its actual checked-out revision, builds and verifies the
ZIP, and uploads `foundation-nano-smoke-package` separately from raw smoke files.
The package retains Python/PyTorch versions in `report.json`; timings and sampled
text may differ across runs/devices. No byte-identical cross-platform promise.

The builder deliberately accepts only the exact authored smoke fixture. It rejects
restricted data, other corpora, inherited checkpoints and incomplete training.
This does not grant redistribution rights for other datasets/models. The verifier
checks bounded ZIP members without extracting paths, file hashes, safe checkpoint
loading, corpus identity, training split identity and report/config/tokenizer
bindings. It does not execute generated code or retrain the model. Hashes detect
corruption, not malicious authorship: an attacker can replace a complete unsigned
package. A digest is not a signature or independent evaluation approval.

Tests train/resume a real random-initialized CPU model and verify round-trip,
exclusive publication, corruption, metadata disagreement, path additions and
private-corpus rejection. Larger redistributable datasets, broader evaluations,
release signing and a verified clean-machine installation remain separate work.
