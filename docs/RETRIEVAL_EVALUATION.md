# Retrieval evaluation

The optional neural backend is evaluated separately from language generation.
The authored fixture in `services/api/tests/fixtures/retrieval/` contains eight
fictional public documents and eight different paraphrase queries (Spanish and
English), with SHA-256 document relevance labels. These records explicitly deny
training permission. They are small diagnostic data, not a representative or
blind benchmark. Do not train on the query labels or tune on these scores and
then describe the same suite as held-out evidence.

`docs/RETRIEVAL_COMPARISON.json` records an actual CPU run using the existing
`artifacts/foundation-smoke-cpu/foundation-nano.pt` checkpoint. Its digest and BPE
identity are in the report; no pretrained external weights were used. At k=3:

| Backend | Recall / hit rate | Mean reciprocal rank |
| --- | ---: | ---: |
| TF-IDF | 0.625 | 0.5000 |
| Own Foundation windowed mean | 0.625 | 0.5417 |

Both retrieve a relevant document for five of eight queries. The small rank
 difference is not evidence of a general improvement. Keep TF-IDF as default.
Scores are at document level over the top k **chunks**: repeated chunks from one
document do not increase recall and can occupy several rank positions. Timing
covers search, including query embedding, but excludes index/model loading.
Index hashes differ because the representations differ; the shared suite hash
allows verification that both ran the same queries and labels.

Reproduce from the repository root with the package installed or
`$env:PYTHONPATH='services/api/src'`. Use fresh output paths:

```powershell
python -m ai_home.knowledge build services/api/tests/fixtures/retrieval/documents.json tfidf.json
python -m ai_home.knowledge build services/api/tests/fixtures/retrieval/documents.json neural.json --embedding-checkpoint artifacts/foundation-smoke-cpu/foundation-nano.pt
python -m ai_home.retrieval_evaluation tfidf.json services/api/tests/fixtures/retrieval/queries.json tfidf-report.json --limit 3
python -m ai_home.retrieval_evaluation neural.json services/api/tests/fixtures/retrieval/queries.json neural-report.json --limit 3 --embedding-checkpoint artifacts/foundation-smoke-cpu/foundation-nano.pt
```

The checkpoint is a local smoke artifact; reproduce its training using the
foundation smoke instructions if unavailable. A different resulting digest must
be reported as a new experiment. The evaluator rejects missing relevance targets,
duplicate IDs, empty labels, exact query/chunk copies and oversized suites. It
cannot establish that externally supplied queries were excluded from training.
Future work needs larger separately authored relevance sets, negative/no-answer
queries, multilingual coverage, trained representation objectives and measured
memory/latency tradeoffs before recommending neural retrieval.
