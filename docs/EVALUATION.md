# Evaluación

Implementado: pérdida de siguiente token y perplexity sobre documentos reservados; tokens evaluados; métricas de entrenamiento y memoria CUDA. La perplexity se limita numéricamente al exponencial de 80 cuando la pérdida lo supera.

Pruebas neuronales escritas: causalidad, equivalencia KV cache, norma de RoPE, gradientes, gradient checkpointing, reanudación exacta CPU, generación, cancelación y smoke CUDA condicionado a hardware. Ejecutadas en CPU; CUDA sigue omitida por la distribución instalada sin CUDA.

No existen todavía resultados de comprensión de instrucciones, español, inglés, código, matemáticas, conocimiento o razonamiento. Una pérdida reducida no demuestra conversación útil. Hace falta un conjunto fijo versionado y ajeno al entrenamiento, evaluación de regresiones y revisión independiente antes de promover candidatos.

Todos los checkpoints nuevos se marcan candidate. No se ha implementado promoción global, comparación de capacidades o rollback centralizado. Seleccionar manualmente un checkpoint local no equivale a aprobarlo para publicación.


## Comparación incremental verificada

`ai_home.foundation.continual` evalúa padre y candidato con el mismo tokenizador,
objetivo y holdout explícito versionado. Registra pérdida, tokens, perplexity,
delta y umbral. Conserva identidades de ambos checkpoints y del conjunto de
evaluación. El control de regresión es solo una comparación de pérdida: no
sustituye benchmarks de capacidades o revisión independiente para publicación.
Véase `CONTINUAL_SMOKE_CPU.json` y `TRAINING.md`. El replay reduce exposición al
olvido pero este experimento pequeño no demuestra que lo elimine.
