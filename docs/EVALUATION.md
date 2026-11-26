# Evaluación

Implementado: pérdida de siguiente token y perplexity sobre documentos reservados; tokens evaluados; métricas de entrenamiento y memoria CUDA. La perplexity se limita numéricamente al exponencial de 80 cuando la pérdida lo supera.

Pruebas neuronales escritas: causalidad, equivalencia KV cache, norma de RoPE, gradientes, gradient checkpointing, reanudación exacta CPU, generación, cancelación y smoke CUDA condicionado a hardware. Ejecutadas en CPU; CUDA sigue omitida por la distribución instalada sin CUDA.

Existen resultados de diez probes autorados en esas categorías, todos fallidos; no constituyen benchmarks representativos de capacidad. Una pérdida reducida no demuestra conversación útil. Hace falta un conjunto fijo versionado y ajeno al entrenamiento, evaluación de regresiones y revisión independiente antes de promover candidatos.

Todos los checkpoints nuevos se marcan candidate. No se ha implementado promoción global, comparación de capacidades o rollback centralizado. Seleccionar manualmente un checkpoint local no equivale a aprobarlo para publicación.


## Comparación incremental verificada

`ai_home.foundation.continual` evalúa padre y candidato con el mismo tokenizador,
objetivo y holdout explícito versionado. Registra pérdida, tokens, perplexity,
delta y umbral. Conserva identidades de ambos checkpoints y del conjunto de
evaluación. El control de regresión es solo una comparación de pérdida: no
sustituye benchmarks de capacidades o revisión independiente para publicación.
Véase `CONTINUAL_SMOKE_CPU.json` y `TRAINING.md`. El replay reduce exposición al
olvido pero este experimento pequeño no demuestra que lo elimine.

## Evaluación de generación

`python -m ai_home.foundation.evaluation --checkpoint candidate.pt --suite docs/evaluation-probes-v1.json --output evaluation.json --device cpu`

La suite exige versión, procedencia, licencia, identificadores únicos, categoría,
prompt y respuestas aceptadas. La puntuación es coincidencia exacta tras NFC y
recorte exterior; no usa un juez externo. No ejecuta código generado. Greedy
(temperatura cero), presupuesto de tokens y hashes se guardan en el reporte.

Se registran respuestas, tokens, tiempo de generación por caso y tokens/s,
memoria CUDA máxima asignada (solo CUDA) y RSS del proceso al terminar. RSS no
es memoria máxima del modelo; tiempo de generación excluye carga del checkpoint.
No contar prompts truncados como éxitos. La comprobación de contaminación
bloquea prompts y pares prompt/respuesta que coinciden con documentos entrenados;
no garantiza ausencia de contaminación semántica. Falta de linaje se declara.

Los probes cubren español, inglés, instrucciones, código, matemáticas,
conocimiento, razonamiento, consistencia y robustez. Son fixtures de mecánica,
no una certificación de esas capacidades. Primera ejecución CPU: todos los
prompts caben en contexto 128, 0/10 aciertos en padre, control y experimento.
Véase EVOLUTION.md para comandos y evidencia conservada.


## Desktop report inspection

The provider control page offers **Ver evaluación o evolución**. Select an
`ai-home-evaluation-v1` report or `ai-home-evolution-result-v1` result. The window
shows the exact probe pass count or experimental decision, followed by the full
report including categories, generated responses, model/suite identities,
configuration and timings when present. Loading is local and read-only; profile
changes close the window and invalidate callbacks. No model promotion occurs.

The bounded reader rejects non-finite numbers, unsupported formats, duplicate
probe IDs, inconsistent category totals and truncation summaries. A report's
file hash identifies its bytes; it does not authenticate who ran it or prove
that the named checkpoint produced it. Evolution reports show recorded control
and experimental categories; this viewer does not rerun the experiment.


## Run an evaluation from the desktop

Choose **Ejecutar evaluación local** on the provider control page, select the
Foundation checkpoint, probe-suite JSON and a new output JSON path. Select CPU
or CUDA and a generation budget (1–4096 tokens per probe), then **Evaluar y guardar**.
The worker executes real inference without blocking Tk and reports completed
probe counts. One evaluation worker can run at a time in this desktop instance.
This does not coordinate GPU resources with separate API/training processes.

Cancel, closing the window, changing profile or closing the app requests
cooperative cancellation. Loading and active kernels cannot be interrupted;
cancellation is checked between generated tokens/probes and before publication.
Incomplete evaluations are not published. Complete reports are flushed and
exclusively linked into place, never overwriting an existing file. Cancellation
after publication does not remove the completed report. No model is activated.


## Identity of evaluated weights

Checkpoint loading hashes the same opened file handle before and after safe
weights loading, with bounded streaming reads. In-place modification during
loading is rejected. The resulting model carries a runtime-only
`checkpoint_sha256`; this attribute is not a model weight or checkpoint field.
Evaluation reports use that identity, not a later read of the pathname. If the
path is replaced after loading, the report still identifies the bytes actually
loaded. Tests change the path between probes and mutate a file during loading.
This is content identity checking, not signature-based publisher authentication.

## Familias declaradas en probes

Cada probe puede incluir `split_group_sha256` (64 hexadecimales en minúsculas).
Antes de generar ninguna respuesta, el evaluador rechaza cualquier grupo que
figure en `training_split_groups` del checkpoint. Esto detecta reformulaciones
de una familia conocida aunque el texto no coincida con documentos entrenados.
La etiqueta se omite al serializar cuando falta, conservando las huellas de
suites anteriores. El informe añade `grouped_probe_count` y
`group_lineage_available` para indicar el alcance de esta evidencia.

No descubre relaciones por sí mismo: preguntas sin etiqueta, familias mal
asignadas y checkpoints anteriores sin linaje de grupos mantienen sus límites.
La protección de coincidencia textual existente sigue activa. Doce pruebas de
evaluación/evolución/comparación aprobadas, incluido rechazo antes de inferencia,
evaluación real de otra familia, validación de formato y serialización anterior.


## Inspección de pérdida distribuida

EVALUATION → Abrir informe admite `ai-home-distributed-loss-report-v1`,
además de evaluaciones generativas y experimentos. Muestra pérdida causal,
tokens, número de fragmentos y el JSON con las identidades del plan, checkpoint
y trabajos. `DISTRIBUTED_EVALUATION_REPORT.json` conserva el informe del smoke
CPU/CUDA local: dos fragmentos y 468 tokens.

El lector comprueba hashes con formato válido, trabajos y documentos únicos,
una misma base entre fragmentos, totales y media ponderada consistente. La
lectura está limitada a 2 MB. Estas comprobaciones no autentican los hashes ni
repiten la ejecución: se conserva el aviso de contenido local no verificado
independientemente. Una pérdida causal no demuestra capacidad lingüística y
abrir el informe no promueve modelos.
