# Public knowledge connectors

Installed-wheel verification before the subsequent cancellation change: wheel SHA256
`dccd5d79ee379d9242c37a1644584ab4cba1d5ead1a02660f5b7a48131d3590b`;
`python -I scripts/verify_knowledge_acquisition.py --require-installed` passed
in the existing isolated CPU environment with controlled HTTP. `pip check`
reported no broken requirements. A separate live CLI request for revision
1385774090 returned HTTP 403; no pending file was created. The earlier live
success remains valid evidence of that execution, not guaranteed availability.
No alternate route, authentication or response bypass was attempted.

Local regression after adding the Wikipedia connector and acquisition dialog:
423 passed, 7 PostgreSQL tests skipped, in 134.71 seconds. This predates the
cancellation change and is not a published revision. The live acquisition metadata in
[WIKIPEDIA_CONNECTOR_SMOKE.json](WIKIPEDIA_CONNECTOR_SMOKE.json) was checked
against the actual pending artifact: sizes, text/source hashes, revision identity
and disabled permissions match.

KNOWLEDGE now offers source acquisition from an explicit JSON manifest to a new
pending JSONL file. It uses the same bounded manifest loader and connector registry
as the CLI, runs in the background, and never approves rights or use permissions.
Cancel, closing the dialog and switching profiles request cooperative cancellation.
Checks run before requests, between decoded response blocks, after each document
and immediately before exclusive publication. An HTTP wait or native parsing call
must finish before cancellation is observed; this is not a hard real-time deadline.
Cancellation after publication does not delete the completed file. Late UI results
are discarded on profile changes. `scripts/verify_knowledge_acquisition.py` exercises
real Tk and the actual connector through controlled HTTP; live Wikipedia access
is separately described below. Visual layout/DPI review remains pending.

Cancellation verification: 25 connector tests passed, including both download
types and cancellation after temporary-file flush. No final or temporary file
remained in those cases. Real Tk also cancelled an active controlled HTTP stream,
restored its start button and published no pending batch.

## Wikipedia revision acquisition

`wikipedia-revision-v1` accepts `connector`, `language`, a positive integer
`revision`, and a declared `license`. It requests that exact revision from the
language Wikipedia API, checks its identity and article namespace, and converts
visible wikitext using the existing parser. It does not expand templates or
render the article. See the [MediaWiki revision API](https://www.mediawiki.org/wiki/API:Revisions).

The existing download/batch limits and review workflow apply. Redirects, hidden
content, API errors and oversized responses fail without publishing a batch.
Records retain page/revision IDs, revision timestamp, pinned URL, response/text
hashes, and page-family grouping compatible with the local dump importer.
Permissions remain false and visibility is pending-review. A declared license
does not establish permission or satisfy attribution obligations.

Live network check: English Wikipedia page 18831, revision 1385774090, acquired
successfully as 58,951 UTF-8 text bytes and a 60,206-byte pending JSONL. Text SHA256:
`30b62e6f231995eecbf06b6d18b3a78d0946bb5c9ff2e2c99dc81528a45b16b1`.
The pending file and unapproved template are retained under
`artifacts/wikipedia-network-v1.*`; neither is admitted to training or retrieval.
This is one source acquisition, not a representative corpus or quality result.

An integration test imports an actual compressed XML fixture through the dump
importer and another revision through the API connector. Different revision/text
hashes retain the same page-family hash. The dataset splitter then refuses to
use those two versions as independent training/validation partitions. This covers
declared page identity across both import paths, not semantic duplication across
different pages. The connector suites currently pass 22 tests.

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

## Complete-file publication

Pending connector batches, review templates, approved datasets and knowledge
indexes now share exclusive complete-file publication. Bytes are written and
flushed to a temporary file in the destination directory before an exclusive
hard link exposes the final name. Existing versions are never replaced, including
when two writers race. Cancellation before publication and ordinary write/flush
failures remove the temporary file and leave no final target. Filesystems without
hard-link support fail explicitly. A forced process termination can leave a
uniquely named `.artifact.tmp` file; it cannot expose a partial final artifact.
This is not a claim of directory-metadata durability across machine power loss.
