# Entrenamiento reproducible

Requiere Python >=3.11, PyTorch >=2.6 y dependencias del proyecto. Instalar con python -m pip install -e ".[dev,llm]" en un entorno con acceso a paquetes. Para CUDA, instalar una distribución de PyTorch compatible con el equipo.

Prueba mecánica completa preparada:

```powershell
python -m ai_home.foundation.smoke --output-dir artifacts/foundation-smoke --device cpu
```

El comando está diseñado para crear un corpus sintético declarado, aprender BPE, inicializar pesos aleatorios, entrenar cuatro pasos, cargar el optimizador y continuar hasta ocho, guardar checkpoints y generar texto real. Es un experimento de mecánica, no un benchmark de calidad. Completado en CPU el 2026-11-26; resultados conservados en `docs/FOUNDATION_SMOKE_CPU.json`. Los checkpoints y el dataset permanecen en `artifacts/foundation-smoke-cpu/` (excluidos de Git). Si el paquete no está instalado, definir `$env:PYTHONPATH="services/api/src"` antes del comando.

Entrenamiento sobre datos propios:

```powershell
python -m ai_home.foundation --dataset dataset.json --output foundation-nano.pt --device auto
```

El dataset es JSON con format_version=1 y records. Cada registro requiere source, url, license, date, language, text, training_allowed=true, privacy_reviewed=true y visibility=public. Los permisos son declaraciones revisadas por el operador, no una verificación jurídica automática. El filtrado detecta patrones comunes de secretos y correos, no todos los datos personales. No conectar directamente conversaciones privadas.

Se separan documentos distintos antes de entrenar el tokenizador. Las secuencias no cruzan documentos; las colas se rellenan con PAD y sus etiquetas se excluyen de la pérdida. La partición se ordena por hash y reserva 20%: añadir documentos puede cambiarla. No sirve todavía como evaluación estable de aprendizaje continuo. Deduplicación semántica y por familias documentales pendiente.

AdamW, warmup/coseno, acumulación, clipping, fp32/fp16/bf16, detección de valores no finitos, métricas y checkpoints periódicos. Resume exige dataset y configuración idénticos y conserva estados del optimizador, escalador y RNG. Usa --stop-after para interrumpir sin modificar el horizonte del scheduler. Cambiar de dispositivo no garantiza reproducibilidad bit a bit.

Las cargas usan weights_only=True, comprobación de dimensiones, finitud y huella del tokenizador. Esto reduce riesgos, pero no convierte checkpoints desconocidos en archivos confiables ni evita todos los agotamientos de recursos. Solo cargar artefactos de origen aprobado. Los archivos de checkpoint no deben añadirse directamente a Git: publicar posteriormente con procedencia y revisión.


## Contribuciones revisadas y SFT

Exportar un dataset congelado desde la API administrativa existente; conservar
el archivo en almacenamiento del operador. El hash comprueba integridad, no
identidad del emisor: solo usar exportaciones de la instancia autorizada.

Crear una configuración JSON de entrenamiento, por ejemplo:
`{"steps": 20, "warmup": 2, "objective": "sft"}` y ejecutar:

```powershell
python -m ai_home.foundation --reviewed-export reviewed.json --training-config sft.json --output artifacts/sft-candidate.pt --device cpu
```

El importador comprueba versión y hash del manifiesto, número de muestras,
consentimiento, revisión humana completa, roles alternos y filtros de contenido.
El consentimiento no autoriza redistribución: los registros y pesos resultantes
se etiquetan `reviewed-training-only`; no se envían a nodos ni se publican.
No se incorpora ningún mensaje directamente desde el chat. Los filtros comunes
no sustituyen la revisión humana ni detectan todos los datos personales.

SFT usa el mismo formato de roles que inferencia. Solo contenido del asistente
y su EOS contribuyen a la pérdida; roles, preguntas y PAD quedan enmascarados.
Las conversaciones se separan antes de entrenar BPE. El checkpoint conserva
versión/hash del dataset, objetivo, métricas y estado candidato, sin almacenar
el texto de las conversaciones. Los pesos pueden memorizar datos: borrar la
exportación no elimina su influencia; una retirada posterior exige reconstruir
versiones afectadas o investigar unlearning, que aún no está implementado.

El comando básico inicializa desde cero o reanuda el mismo entrenamiento. El
módulo continual permite fine-tuning desde otro candidato con replay y una
evaluación explícita fija. La promoción aprobada sigue pendiente. Esta ruta demuestra
SFT mecánico, no capacidad conversacional. El empaquetado ahora incluye colas:
no comparar métricas históricas como si fueran la misma configuración de datos.
La nueva configuración incluye `objective`; resume rechaza configuraciones
antiguas sin ese campo en lugar de asumir equivalencia.

## Aprendizaje incremental controlado

```powershell
python -m ai_home.foundation.continual --parent parent.pt --output-dir artifacts/candidate-v2 --validation holdout.json --replay previous.json --dataset incremental.json --training-config training.json --device cpu
```

Alternativamente usar `--reviewed-export reviewed.json` y objetivo `sft`.
Replay y holdout deben ser corpus públicos revisados; para SFT deben contener
conversaciones. Replay de exportaciones restringidas todavía no está conectado.
Todos los datasets públicos usan el esquema versionado de `--dataset`.

El padre debe tener hashes de documentos de entrenamiento/evaluación. Los
checkpoints antiguos sin trazabilidad se rechazan. El candidato conserva
arquitectura y BPE del padre; usa un optimizador nuevo. Resume conserva el
optimizador del candidato y exige datasets/configuración idénticos.

Cada lote contiene ejemplos nuevos y replay (25% solicitado, redondeado a
muestras enteras con al menos una de cada clase). El informe registra el número
real de muestras replay. Se rechaza contaminación exacta con el holdout actual
y con todos los holdouts de la cadena de ancestros; no detecta paráfrasis ni
contaminación semántica. El holdout de regresión no debe reutilizarse para elegir
repetidamente hiperparámetros: reservar también una evaluación final independiente.

La salida exige un directorio nuevo y nunca sustituye al padre. El reporte
compara pérdida sobre exactamente los mismos tokens, registra hashes y marca
el resultado candidato. Superar el umbral no autoriza promoción. Las restricciones
de uso del padre se propagan aunque los datos incrementales sean públicos.

Experimento reproducible ejecutado en CPU:

```powershell
python -m ai_home.foundation.continual_smoke --output-dir artifacts/continual-smoke-v2 --device cpu
```

Usar un directorio inexistente. Guarda corpus sintéticos declarados, configuraciones,
padre, candidato, checkpoints intermedios y reporte. Evidencia conservada en
`docs/CONTINUAL_SMOKE_CPU.json`: pérdida 5,621232 a 5,580207 sobre 1.536 tokens,
ocho pasos iniciales y ocho incrementales. No demuestra calidad conversacional,
control general de catastrophic forgetting ni mejora de razonamiento.

## Entorno GPU verificado

Usar `.venv-cuda/Scripts/python.exe` para entrenar con CUDA en este equipo; el Python habitual sigue usando PyTorch CPU. La configuración probada y los comandos completos están en CUDA_VALIDATION.md. FP32/FP16/BF16 están verificados en la RTX 4070 SUPER con modelos diminutos; no presupone tamaños mayores ni participación de cualquier GPU.


## Explicit resource fitting

The Foundation CLI accepts `--auto-fit` for fresh training. It compares a shared
heuristic memory estimate against the smaller of the configured budget and 80%
of current free GPU memory (CUDA) or available RAM (CPU). It reduces microbatch
size using divisors of the original effective batch, increasing accumulation to
preserve that effective batch size. This preserves sample count, not a guarantee
of bit-identical optimization.

If even one sample cannot fit, the default is an explicit error. Add
`--allow-model-shrink` to authorize reducing layers, then width/hidden size while
preserving vocabulary, attention divisibility and context. The requested model is
the upper bound; there are no unmeasured Nano/Micro/Small capacity claims. This
order is a deterministic fitting policy, not an architecture-quality search.
Automatic fitting is prohibited with `--resume`; use the recorded configuration
for reproducible resume. Fine-tuning/replay do not yet use this CLI fitting path.

```powershell
python -m ai_home.foundation --dataset authorized.json --model-config model.json --training-config training.json --device cuda --auto-fit --output artifacts/fitted.pt
```

The command exclusively creates `<checkpoint>.resource-plan.json` with requested
and selected configurations, available/effective budgets and the estimate. A
failed run may leave this plan for diagnosis; choose a new output path for a new
attempt. The normal trainer now shares the same estimator and checks available
CPU RAM as well as GPU memory. Final tokenizer vocabulary may be smaller than the
requested ceiling; the actual checkpoint configuration is authoritative.

Calibration exposed a missing runtime allowance: a tiny CUDA experiment estimated
704,640 bytes but measured 17,712,128 bytes allocated. The estimator now includes
64 MiB runtime/optimizer workspace allowance; the repeated run estimated
67,813,504 bytes and measured 17,712,128. `RESOURCE_PLAN_CUDA.json` records both the
initial discrepancy and the corrected plan, real metrics and parameter change.
The actual CLI with auto-fit also trained and wrote its checkpoint/plan on CUDA.

This is a heuristic, not a hard memory sandbox or measured maximum-fit guarantee.
Other kernel workspaces, allocator fragmentation, concurrent processes and host
dataset tensors can exceed the estimate. No automatic OOM retry, live batch
resizing, multi-GPU sizing, or large-model benchmark has been demonstrated. The
runtime allowance is calibrated on one tiny case, not proven sufficient at scale.


## Controlled cancellation

The training CLI handles the first Ctrl+C by requesting a controlled stop. After
the current accumulated optimizer update and validation, it saves both the
numbered checkpoint and output checkpoint, including optimizer, scaler and RNG
state. Resume with the same configuration and `--resume` as usual. The Python
trainer accepts a threading-compatible `cancel` event for the same behavior.
Checkpoints and the desktop inspector expose `training_complete` and
`stop_reason` (completed, cancelled, step_limit or periodic_checkpoint).

Cancellation before the first update raises `TrainingCancelled` without writing
a new checkpoint; the CLI exits 130. A second Ctrl+C interrupts immediately and
may prevent a fresh checkpoint. Cancellation is cooperative: it cannot interrupt
a running CUDA kernel or validation, and a request arriving during checkpoint
saving may be observed after the next optimizer update. Completing the final
planned step is reported as completed even if a stop was also requested.

CPU and CUDA tests stop real training after step 2 with gradient accumulation,
reload the saved checkpoint and resume to step 4. Every resulting model tensor
is compared against uninterrupted training (atol 1e-6, rtol 1e-5).


## Fresh training from a parent candidate

Use `--initial-checkpoint parent.pt` to preserve a parent model and tokenizer
while starting a new optimizer and training schedule. This differs from exact
`--resume`; the flags are mutually exclusive. `--validation-dataset holdout.json`
uses an explicit authorized independent holdout. Existing lineage checks reject
training on earlier holdouts or evaluating earlier training documents.

```powershell
python -m ai_home.foundation --dataset corpus.json --validation-dataset holdout.json --initial-checkpoint expanded.pt --training-config training.json --output trained.pt --device cuda --auto-fit
```

With a parent, automatic resource fitting uses the actual parent architecture
and may adjust microbatch/accumulation. Architecture shrinking is prohibited.
An explicitly supplied model configuration must match the parent. The output
is a candidate requiring evaluation and approval, not a newly active model.


## Desktop training control

On the provider page choose **Entrenar modelo local**. Select an authorized public
dataset JSON, an optional model configuration JSON, a required training
configuration JSON and a new run directory. Choose CPU or CUDA. To resume, select
the prior checkpoint and retain the exact training configuration/dataset; leave
model configuration blank to use the saved architecture. Without a parent or
model configuration, the trainer uses its default architecture.

The worker displays actual optimizer steps, training loss and throughput. The
new run directory retains `configuration.json`, numbered checkpoints, `model.pt`
and, after a successful or controlled-stop return, `result.json`. Existing run
directories are rejected. **Parar y guardar** requests cancellation at a complete
optimizer update; no new checkpoint is produced if stopped before the first
update. Closing the dialog or changing profile also requests cancellation.
Normal application close waits for the training worker to finish saving; forced
process termination cannot guarantee a new checkpoint.

Desktop evaluation and training workers exclude one another within this app
instance. Other processes are not coordinated. This initial training form accepts
public datasets; reviewed private SFT exports and fresh-parent fine-tuning remain
available through the CLI. It does not automatically fit resource budgets or
activate a checkpoint; configuration limits and trainer validation apply.
