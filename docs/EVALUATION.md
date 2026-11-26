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
