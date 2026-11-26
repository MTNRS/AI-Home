# Consented corrections for training candidates

An authenticated owner can submit a corrected assistant reply through
`POST /api/v1/training/conversations/{contribution_id}/corrections`.
The request includes a new `conversation_id` UUID, zero-based `message_index`,
`replacement` text, `consent: true`, `consent_version: training_consent_v1`,
`consent_scope: conversation_for_aihome_training_candidates`,
`provider_terms_allow_reuse: true`, and
`content_reviewed_for_sensitive_data: true`.

The endpoint can only read the current account's existing contribution. It does
not collect an unsubmitted private chat. It requires renewed consent and both
attestations, targets an assistant message, rejects unchanged/blank replacements,
and applies the existing credential filter to the resulting conversation.
The new candidate contains the original prefix ending at the corrected answer;
later turns are excluded because they may depend on the old answer. The original
candidate and any frozen dataset containing it are not rewritten.

The correction starts `pending_review` regardless of the parent's status. Stored
provenance records `human_correction_v1`, the source contribution ID, message index
and SHA-256 of the replaced text. Provider/model fields retain the original chat
provenance; the correction marker explicitly identifies the human modification.
No extra copy of the replaced response is stored in correction metadata. This
hash is an audit identifier, not encryption or guaranteed anonymization.

Independent review, dataset freeze and restricted SFT export remain mandatory.
The frozen manifest binds correction metadata, and the import pipeline preserves
it. Existing exports without correction metadata remain readable; null fields are
omitted from API dataset export so legacy manifest bytes/semantics are preserved.
Corrections inherit the normal account deletion/withdrawal rules of contributions;
withdrawing a parent is not an automatic withdrawal of a separately consented
correction. Neither action guarantees removal of influence from trained weights.

Apply migration `0009_training_corrections` before serving this API on an existing
database. It adds a nullable JSON column and leaves prior rows unchanged. No
production database was migrated during implementation. SQLite upgrade, schema
comparison and downgrade are exercised locally; PostgreSQL coverage runs in CI.

Tests cover cross-account rejection, consent/rights/privacy denials, invalid roles,
secret filtering, a distinct pending candidate, unchanged original, duplicate
submission, review gates, frozen hash/export, and real restricted SFT parameter
updates. This is a correction-to-SFT pipeline, not preference optimization or proof
that a suggested correction is true. UI editing and preference-pair learning remain
separate pending work. Initial contribution providers retain their existing limits.

## Contributions from the own Foundation model

The contribution API also accepts `provider: ai-home-foundation`. In that case,
`model` must be the lowercase 64-character SHA-256 of the checkpoint actually used
and `source_endpoint` must equal `ai-home-foundation://sha256/<that digest>`.
Mutable model filenames and HTTP URLs are rejected for this provenance type.
Consent and rights/privacy attestations remain mandatory, and the candidate still
requires independent review. The retained `provider_terms_allow_reuse` field is a
reuse-rights attestation for this source too; own weights do not automatically
authorize redistribution or training on every private conversation.

A test trains an own checkpoint, generates a real reply using the loaded model,
submits that reply with the loader's bound checkpoint digest, and creates a human
correction through the same restricted pipeline. The server validates the format
and linkage; it cannot prove that an arbitrary client used the claimed weights.
The supplied digest is provenance to review, not remote execution attestation.

The desktop local Foundation path now carries the digest bound by the checkpoint
loader with its generated answer. It pins that digest across turns and rejects a
changed checkpoint before generating the next answer. Consented submission uses
that digest rather than a filename or a file reread after generation. Clearing
the chat clears the pin. Server-proxy chats remain excluded until equivalent exact
provenance is available; they are not silently relabeled.
