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
