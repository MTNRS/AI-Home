# Knowledge retrieval

`ai_home.knowledge` implements a bounded local public-source index and connects
retrieval to the real Foundation inference path. It does not train model weights.
The current embeddings are sparse TF-IDF vectors built from indexed text, with
cosine similarity; they are not learned semantic embeddings or a pretrained model.

## Build and inspect

Use the existing Foundation dataset JSON format (`format_version: 1`, `records`).
Each record requires source, HTTP(S) provenance URL, license, date, language, text,
`retrieval_allowed: true`, `privacy_reviewed: true`, and `visibility: public`.
`training_allowed` is independent and may be false for retrieval-only material.
These permissions are operator attestations: software cannot determine a license
or prove the absence of all personal data. The existing reviewed-public dataset
validator and additional metadata filtering reject common credentials/emails.
Private memory and consented conversation exports are not index inputs, even
when a record claims retrieval permission. The training loader and train/validation
split retain their training-only default and reject retrieval-only records.

```powershell
$env:PYTHONPATH = "services/api/src"
python -m ai_home.knowledge build authorized-dataset.json knowledge-v1.json
python -m ai_home.knowledge search knowledge-v1.json "consulta"
```

The build command creates a new file exclusively, normalizes/deduplicates records,
and hashes the versioned payload. It never overwrites an existing index. Publish
an update at a new path and switch the configured path to select that version.
The reader checks integrity, permissions and the 8 MB budget before indexing.
A hash detects accidental modification; it does not authenticate a malicious
operator. Index paths are server configuration, not supplied by chat users.

## Foundation API

Set `AI_HOME_KNOWLEDGE_INDEX` and `AI_HOME_FOUNDATION_CHECKPOINT`, with the
Foundation backend selected. An authenticated request to
`/api/v1/chat/completions` can explicitly set `use_knowledge: true` alongside
`messages`. Retrieval runs locally; the query is not sent to a search service.
The response includes `choices` plus `references`, each recording source URL,
license, date, language, text, offset, document hash, score and index version.
References describe evidence supplied to generation, not verified support for
whatever the model actually says. Empty retrieval yields an empty reference list.
No automatic external-provider fallback occurs.

Retrieved source strings enter only user-level context. Existing system messages
are unchanged. Personal memories likewise use user-level context. This prevents
privileged-role assignment; it does not prove that an untrained model resists all
prompt injection. Sources remain untrusted.

The checkpoint tokenizer measures the complete input before generation. Whole
retrieved chunks are admitted only when they fit. If matching evidence exists but
none fits, or the conversation itself is too long, generation fails explicitly
(the HTTP API returns a generic 503 without leaking local paths or private data).
The normal non-RAG generation path keeps its existing context behavior.

## Evidence and remaining work

`test_knowledge.py` covers deterministic retrieval, immutable versions, tamper
rejection, private/unauthorized inputs, provenance, source-role separation,
context limits and an authenticated API request using actual generation from a
newly trained random-initialized small Foundation checkpoint. The corpus is an
authored synthetic fixture. Generated text is not a demonstrated factual answer.

There is no network ingestion, online refresh scheduler, semantic neural encoder,
or large-scale vector database in this increment.
Existing Wikipedia ingestion remains separate. Arbitrary source connectors,
review workflows and semantic retrieval evaluation remain
part of the full objective. The dataset validator distinguishes retrieval from training permission. Legacy
records without `retrieval_allowed` retain the prior public-training permission
behavior for compatibility. Any explicit retrieval denial takes precedence.
New source integrations should always supply both permission flags.


## Desktop

Under provider routing, enable the server and explicitly select retrieval of
reviewed public sources. Local direct-provider mode rejects this option rather
than silently ignoring it. The setting resets when the account changes.
The chat's “Fuentes recuperadas” button opens a read-only viewer grouping sources
by assistant message, with URL, license, date, language, document hash, offset,
index version and the actual excerpt. No URLs are opened automatically.
The completion status reports the number of sources, including zero matches.
References are kept outside the conversation messages and cleared with the chat;
private viewers close when the account changes. The viewer explicitly states that
retrieved sources do not verify the answer.

Until training exports can retain and review source licenses, the desktop blocks
contribution submission for conversations that used retrieval. This is a client
workflow safeguard, not server-side provenance enforcement for arbitrary clients.
The existing explicit-consent review process still applies to all submitted data.

Validation: 175 tests passed, 4 PostgreSQL tests skipped. A real Windows Tk
construction check (windows hidden, settings/memory loaders isolated) opened the
reference viewer and exercised chat clearing and profile switching successfully.
This is a widget/function check, not a visual layout review.


Permission separation validation: the loader preserves `training_allowed: false`
through indexing, rejects retrieval-only data through every training entry path,
and accepts only the literal boolean true for retrieval. Version update tests
prove that old index bytes remain unchanged and the new version retains its own
source URL, date and document hash. Revoking access requires selecting an updated
index or removing the old index from service; versioning itself is not revocation.
