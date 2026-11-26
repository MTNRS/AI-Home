# AI@Home Evolution Engine

Implementado un primer ciclo experimental acotado a la tasa de aprendizaje.
`ai_home.foundation.evolution` evalúa el padre, detecta la categoría con menor
acierto, registra una hipótesis y crea configuraciones de control/experimento.
Entrena dos candidatos reales desde el mismo padre, datos, replay, semilla y
presupuesto; cambia únicamente learning_rate. Guarda ambos checkpoints,
propuesta previa, evaluación del padre, evaluaciones de candidatos y conclusión.

Solo propone aceptación si mejora estrictamente la categoría objetivo frente
al control, no empeora ninguna categoría frente al control ni al padre y la
pérdida reservada no supera al control ni al padre. La propuesta exige revisión
independiente: no modifica el modelo activo, permisos, privacidad, créditos ni
código. No ejecuta programas generados. Actualmente se prueba una hipótesis
predefinida; no hay investigador autónomo general ni búsqueda arquitectónica.

## Experimento CPU ejecutado el 2026-11-26

Corpus sintético propio, modelo de contexto 128, ocho pasos por candidato,
learning_rate de control 0,0003 y experimental 0,00015. Diez pruebas autoradas
en nueve categorías. Todos los prompts caben; ambos candidatos obtienen 0/10.
Pérdida del control: 5,554141. Pérdida experimental: 5,583737.
**Resultado: rechazo.** No se demostró mejora de capacidades ni se promovió nada.

Informe: `EVOLUTION_SMOKE_CPU.json`. Respuestas/tiempos/memoria por modelo:
`EVALUATION_BASELINE_CPU.json`, `EVALUATION_CONTROL_CPU.json` y
`EVALUATION_EXPERIMENT_CPU.json`. Artefactos locales en
`artifacts/evolution-experiment-v1/`; no se añaden pesos a Git.

## Reproducción

Usar directorios nuevos (los comandos rechazan sobrescribir experimentos):

```powershell
$env:PYTHONPATH="services/api/src"
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
python -m ai_home.foundation.continual_smoke --output-dir artifacts/evolution-inputs-v2 --device cpu --context 128
python -m ai_home.foundation.evolution --parent artifacts/evolution-inputs-v2/parent.pt --output-dir artifacts/evolution-experiment-v2 --suite docs/evaluation-probes-v1.json --validation artifacts/evolution-inputs-v2/validation.json --replay artifacts/evolution-inputs-v2/base.json --dataset artifacts/evolution-inputs-v2/incremental.json --training-config artifacts/evolution-inputs-v2/training-config.json --device cpu --max-new-tokens 32
```

El informe está ligado a hashes de pesos, tokenizador, suite y datasets. Las
mediciones temporales dependen del equipo. Estas pruebas no son un benchmark
representativo; usar además un conjunto final independiente, más muestras y
múltiples semillas antes de concluir una mejora. La selección repetida por una
suite puede sobreajustarla. El control de contaminación detecta coincidencias
exactas de documentos, no paráfrasis o solapamientos parciales.

Aprendizaje incremental y SFT están descritos en TRAINING.md. Promoción aprobada,
rollback centralizado, experimentos arquitectónicos y evaluaciones amplias
continúan pendientes. La eliminación del archivo fuente no borra su influencia
en pesos entrenados; no se promete unlearning exacto.


## Individual regression gate

Evolution decisions now require matching suite, generation configuration, device
and PyTorch version, with identical probe IDs/categories and totals consistent
with individual outcomes. Any previously passing probe that fails in the
experiment rejects acceptance, even if the category accuracy rises overall.
This check applies against both the control and the parent baseline. Result
reports record `control_probe_regressions` and `parent_probe_regressions`. Both
control and experimental held-out losses must be finite. Independent review
remains required; no experimental decision activates a model.


## Controlled experiment interruption

`run_experiment` accepts `cancel` (threading-compatible event) and `on_progress`.
Progress identifies baseline evaluation, control training/evaluation and experiment
training/evaluation, with actual step or probe metrics. The evolution CLI prints
these events. First Ctrl+C requests a controlled stop; second Ctrl+C interrupts
immediately and may prevent checkpoint/report persistence.

Training stops after a complete optimizer update and retains its resumable
checkpoint. Inference checks cancellation between tokens/probes. Held-out loss
comparison checks before/after each comparison; it cannot interrupt an active
comparison/kernel or checkpoint load. An interrupted evolution run records its
phase in `interrupted.json`, leaves completed artifacts intact and does not write
`result.json` or proceed to the next experiment. Cancellation before starting
creates no run directory. The CLI exits 130 on cooperative cancellation.

The complete experiment is not automatically resumable as an orchestrated job;
retained training checkpoints can be resumed with the existing Python training
API using matching data/replay settings. Partial work is not an acceptance decision.


## Desktop experiment control

**Ejecutar evolución local** selects a parent checkpoint, probe suite, independent
validation corpus, replay corpus, incremental dataset (or reviewed SFT export),
training configuration and new output directory. Choose CPU/CUDA, learning-rate
multiplier and probe token budget. The form executes the existing controlled
experiment, displays real stage/step/probe progress and records the final proposal.
It cannot activate a model.

Training, evaluation and evolution exclude one another within this desktop
instance. Closing the experiment window or changing account requests cancellation;
normal application close waits for controlled shutdown. Other applications and
API processes are not coordinated. Output directories are exclusive, and the
normal lineage, authorization and regression gates remain in force.

### Secuencias locales en disco

El CLI admite `--disk-sequences`; `run_experiment` y `run_evolution` admiten
`disk_sequences=True`. Se aplica por igual a control y experimento, incluyendo
entrenamiento, replay y comparaciones de perdida. La propuesta y ambos informes
registran la opcion. Cada rama conserva sus caches separados dentro de su
subdirectorio, sin reutilizar archivos preexistentes. La evaluacion generativa
por probes conserva su recorrido habitual.

No cambia las reglas de aceptacion ni publica pesos. Los caches pueden contener
datos privados tokenizados y no deben tratarse como artefactos publicos. La opcion
reduce materializacion de secuencias en RAM, sin garantizar una cota estricta.
El dialogo de evolucion incluye la casilla «Guardar secuencias en disco»,
desactivada por defecto. Su valor se captura antes de lanzar el trabajador.
El valor predeterminado conserva la preparacion en memoria.
