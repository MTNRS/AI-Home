# Pinned repository-document training experiment

Source revision: `2724cf7d96d90a11d9665bd9dcb64c920d629af1` of AI-Home,
whose project license is Apache-2.0. Reviewed technical excerpts were selected
explicitly: first 100 lines of FOUNDATION_MODEL and 85 of TOKENIZER for training;
first 75 lines each of TRAINING and EVALUATION for validation. These include
historical implementation statements, not uniformly current factual guidance.
No Wikipedia content or private conversations were used.

The script saves normalized/versioned datasets, original-file hashes, excerpt
bounds, source URLs and file-family identifiers. BPE learns only from training
files. Validation files are held apart, but their subject matter overlaps heavily;
this is not an independent general-language benchmark or a representative corpus.

Reproduce with `scripts/train_repository_documents.py --revision
2724cf7d96d90a11d9665bd9dcb64c920d629af1 --output-dir NEW_DIRECTORY --device cuda`
using the project Python environment. The benchmark Nano configuration has
803,968 parameters, 512 vocabulary slots and 256 context positions. All weights
start randomly. Training uses BF16, seed 42, 200 steps and the existing resource
guards; a one-step checkpoint is retained and resumed into a separate candidate.

Observed validation loss fell from 6.256052350614601 after one step to
5.208435115646616 after 200, over the same 5,349 held-out tokens. Embedding L2
change since step one was 3.3768553733825684. The one-step comparator is not an
untrained baseline. Greedy generation was repetitive and incoherent; no model
was promoted and no conversational capability is established.

Checkpoints, complete metric history, datasets and configuration remain under
`artifacts/repository-documents-v1/`. A compact result is preserved in
`REPOSITORY_DOCUMENT_EXPERIMENT.json`. This experiment adds evidence about the
training pipeline; it does not resolve the need for diverse authorized corpora,
independent evaluation or broader model capabilities.

The script now reads the Nano model configuration from the same immutable commit
as the documents and records its source SHA256. `--prepare-only` writes the
datasets and configuration without loading/training a model. This preparation
was executed separately and matched both original dataset files byte for byte,
plus every model/training setting from the actual experiment. It produced no
checkpoint. This confirms that the original run used the pinned configuration;
the initial script had read that unchanged file from the working tree instead.
Runtime code and dependency versions still need to be preserved separately for
exact reproduction; pinning input data is not a guarantee of bitwise GPU results.

## Generation comparison

Both checkpoints were evaluated on `docs/evaluation-probes-v1.json`, using CUDA,
greedy generation and at most 64 new tokens. All ten prompts fit the context.
Both scored **0/10**. The lower held-out causal loss therefore did not translate
into a measured improvement on these authored exact-answer probes. This small
suite is itself a mechanical fixture, not a representative capability benchmark.

Full answers and timings are retained in
`REPOSITORY_DOCUMENT_ONE_STEP_EVALUATION.json` and
`REPOSITORY_DOCUMENT_CANDIDATE_EVALUATION.json`. The report reader validated both
documents and their checkpoint hashes were compared with the actual loaded files.
Reproduce each using `python -m ai_home.foundation.evaluation --checkpoint PATH
--suite docs/evaluation-probes-v1.json --output NEW_REPORT --device cuda
--max-new-tokens 64`. No model was promoted.

## Tokenizer efficiency

The learned 512-slot BPE was compared with the 262-slot byte baseline (256 bytes
and six reserved tokens) using the exact retained document partitions. Both
encodings round-tripped every normalized text. Training: 10,389 UTF-8 bytes became
4,847 tokens, a 53.34% token-count reduction. Validation: 10,402 bytes became
5,347 tokens, a 48.60% reduction. These counts exclude BOS/EOS/PAD and conversation
wrappers, explaining their difference from causal-loss token counts.

`scripts/measure_repository_tokenizer.py` requires exact document-hash agreement
with the checkpoint's training and validation lineage before publishing its
immutable report. Swapping the partitions was tested and rejected. Full counts
and identities are in `REPOSITORY_TOKENIZER_EFFICIENCY.json`. This is segmentation
efficiency on four related technical excerpts, not measured throughput, broad
multilingual coverage or language understanding.
