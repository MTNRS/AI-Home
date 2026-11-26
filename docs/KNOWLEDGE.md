# Knowledge retrieval

`ai_home.knowledge` implements a bounded local public-source index and connects
retrieval to the real Foundation inference path. It does not train model weights.
The current embeddings are sparse TF-IDF vectors built from indexed text, with
cosine similarity; they are not learned semantic embeddings or a pretrained model.

## Build and inspect

Use the existing Foundation dataset JSON format (`format_version: 1`, `records`).
Each record requires source, HTTP(S) provenance URL, license, date, language, text,
`training_allowed: true`, `privacy_reviewed: true`, and `visibility: public`.
These permissions are operator attestations: software cannot determine a license
or prove the absence of all personal data. The existing reviewed-public dataset
validator and additional metadata filtering reject common credentials/emails.
Private memory and consented conversation exports are not index inputs.

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
large-scale vector database, or desktop reference panel in this increment.
Existing Wikipedia ingestion remains separate. Arbitrary source connectors,
review workflows, semantic retrieval evaluation and desktop presentation remain
part of the full objective. The current public training permission requirement is
stricter than retrieval-only licensing and needs a distinct permission model in
a later pipeline revision.
