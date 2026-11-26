# Public knowledge connectors

`ai_home.knowledge_connectors` downloads only explicitly listed source documents.
The initial `github-text-v1` connector accepts public repository text at a full
40-character commit SHA. It accesses the fixed `raw.githubusercontent.com` host,
sends no authentication credentials, ignores environment proxy configuration,
does not follow redirects, and does not execute downloaded content. Supported
suffixes are `.md`, `.txt`, `.rst`, `.py`, `.json`, and `.csv`; this does not imply
special parsing for each format. All are treated as UTF-8 text.

Limits: 20 files per manifest, 500,000 decoded response bytes per file, 8 MB for
the pending JSONL, 64 KiB manifest, 20-second HTTP timeouts. There is no crawler,
repository enumeration, branch-head tracking, authentication or automatic retry.
HTTP denial/rate limits fail the batch; they are not bypassed. Commit pins make
updates explicit: change the pin in a new manifest and produce a new output path.
The source license string is a declaration, not verified legal permission.

Each pending record includes commit/path, source URL, access date, language,
raw-byte SHA-256 and normalized-text SHA-256. Normalization is NFC, CRLF to LF,
and edge whitespace removal, matching dataset ingestion. Both use permissions
and privacy review are false. Public availability never grants training rights.
The existing review workflow binds the full record metadata and text hash before
approving retrieval or training separately. Operators must inspect source rights,
privacy and relevant exceptions before approving; no automatic approval occurs.

```powershell
$env:PYTHONPATH='services/api/src'
python -m ai_home.knowledge_connectors configs/knowledge/github-example.json pending.jsonl
python -m ai_home.knowledge_ingest template pending.jsonl review.json
# Inspect documents and rights; explicitly complete per-document review fields.
python -m ai_home.knowledge_ingest approve pending.jsonl review.json approved.json --purpose retrieval
python -m ai_home.knowledge build approved.json index.json
```

The review template starts with every permission disabled and no reviewer. The
approval command fails until an actual review is recorded. The pipeline requires
at least two distinct approved documents; a single-file acquisition can be held
pending until combined with another reviewed source.

`CONNECTORS` is an extension registry for installed Python implementations.
Manifest values cannot import arbitrary modules or run plugins from the network.
Future scientific/public-domain/technical-data connectors must preserve the same
pending-review contract and have source-specific access and licensing validation.
This first implementation does not cover that full source range.

Real network validation downloaded README and architecture documentation from the
public AI-Home repository at `2ddea7d`, totaling 24,891 pending JSONL bytes. See
`GITHUB_CONNECTOR_SMOKE.json` for hashes and URLs. The files and an unapproved
review template are under ignored `artifacts/github-connector-v1/`; neither was
admitted to training or retrieval. Tests also cover review promotion, denied use,
immutable references, invalid paths, redirection, size limits and HTTP failures.
