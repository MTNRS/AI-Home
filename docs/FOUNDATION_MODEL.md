# AI@Home Foundation Model

Estado: implementación inicial local, mecánica neuronal validada en CPU. No es una entrega completa del objetivo Foundation Model.

El paquete services/api/src/ai_home/foundation separa arquitectura, tokenizador, datos, checkpoints, entrenamiento e inferencia. Todos los parámetros se inicializan aleatoriamente; no importa pesos ni tokenizadores de terceros. El GPT byte-level anterior se conserva.

Arquitectura: decoder causal, RoPE, GQA configurable, RMSNorm, SwiGLU, conexiones residuales, embeddings compartidos opcionales, KV cache y gradient checkpointing. Nano por defecto usa anchura 128, cuatro capas y contexto 256. Es una configuración conservadora de arranque, no una elección óptima demostrada en la RTX 4070 Super. El smoke usa anchura 32 y dos capas para verificar mecánica.

Micro, Small, Medium y Large siguen pendientes de benchmarks y presupuestos medidos. No se atribuye capacidad lingüística, razonamiento o multimodalidad. La estimación de memoria es aproximada y rechaza configuraciones excesivas; todavía no realiza autoajuste mediante benchmarks.

El proveedor de escritorio AI@Home Foundation acepta el nombre de un checkpoint .pt del directorio de modelos de AI@Home. No tiene fallback a un proveedor externo. El proveedor byte-level mantiene su comportamiento previo. El servicio HTTP expone Foundation mediante `/api/v1/chat/completions`, con autenticación y checkpoint configurado por el operador. La prueba de integración entrena un checkpoint y compara las respuestas HTTP con inferencia directa real. La interfaz gráfica sigue sin revisión visual.

Referencias técnicas consultadas: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention y https://docs.pytorch.org/docs/stable/notes/serialization . La implementación utiliza primitivas de PyTorch y no incorpora una implementación completa de otro modelo.



## Local checkpoint inspection

The desktop provider section includes “Inspeccionar checkpoint local”. Select an
existing `.pt` file to open read-only Model and Training tabs. The same inspection
is available as:

```powershell
python -m ai_home.foundation.inspection artifacts/model.pt
```

Inspection uses the safe Foundation checkpoint loader on CPU and a default 2 GB
file/model budget. It verifies the file hash before/after loading, derives the
parameter count from the validated model, verifies tokenizer identity and exposes
architecture/configuration plus retained training metrics and validation loss.
A declared parameter-count mismatch is flagged rather than used as the displayed
count. It omits model tensors, optimizer state and individual training-document
hash lists. The view is local and does not upload the checkpoint, select it for
chat or promote it in the registry. It closes on profile changes.

Training metrics are checkpoint metadata, not an independent capability audit.
Checkpoint `status` is explicitly distinct from registry approval. Inspection
currently materializes the model on CPU; large checkpoints may be expensive even
within the file budget. Metadata must serialize to finite JSON within a separate
2 MB limit. The technical views show structured values; interactive plots and
training controls remain future interface work.

Validation trains a real checkpoint, compares reported metrics and architecture,
checks a tampered declared count against the actual parameter count, and rejects
oversized/corrupt files. A hidden Windows Tk check opened the actual CUDA resource
experiment checkpoint (27,552 parameters) and verified both tabs and profile
closure. Full suite: 230 passed, 4 PostgreSQL skipped.

### Experimental own-model embeddings

`ai_home.foundation.embeddings.FoundationEmbedder(checkpoint, device="cpu")`
loads the project's own checkpoint and exposes `embed(text)` and `identity`.
It averages final normalized causal Transformer states over every token in
non-overlapping context windows, then L2-normalizes the vector. Windows reset
positions; there is no cross-window attention, special-token insertion, padding,
or silent truncation. Inputs must be nonempty and at most 32,000 UTF-8 bytes.
The identity includes the digest of the checkpoint bytes actually loaded, the
BPE fingerprint, dimensions, window size, and pooling protocol version.
CPU and explicit CUDA are supported; no external weights are used.

This is an experimental feature extraction primitive, not a trained semantic
encoder. TF-IDF remains the default. The optional neural index described below provides
checkpoint-bound retrieval. Independent retrieval-quality comparisons remain
necessary before recommending it. No semantic-quality gain is claimed.

### Optional neural knowledge index

Build a separate immutable knowledge index with the own checkpoint:

```powershell
python -m ai_home.knowledge build approved-dataset.json neural-index.json --embedding-checkpoint model.pt --embedding-device cpu
python -m ai_home.knowledge search neural-index.json "consulta" --embedding-checkpoint model.pt
```

The public retrieval permission and privacy review rules still apply, even if
training permission is false. Index version hashing covers vectors, checkpoint
identity, and source records. Loading requires matching checkpoint digest, BPE,
window size, dimensions and pooling method; malformed/non-normalized vectors are
rejected. The bounded reference index permits at most 200,000 vector components
and retains the existing 8 MB serialized limit. Updates require a new filename.
RAG accepts this index with that same generation checkpoint, reusing the loaded
model. Source references and whole-chunk context fitting are preserved. Different
embedding/generation checkpoints are not supported in this initial integration.

Without `--embedding-checkpoint`, building and searching retain TF-IDF behavior.
Neural search ranks all chunks by cosine, including nonpositive similarities;
it has no calibrated relevance threshold. Exact-text self-retrieval tests verify
the mechanism, not semantic generalization. Independent relevance benchmarks
are still required before recommending this backend over TF-IDF.
