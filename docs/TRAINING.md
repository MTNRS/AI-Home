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
instance. Other processes are not coordinated. The form does not automatically
fit resource budgets or activate a checkpoint; configuration limits and trainer
validation apply.

The source selector also accepts an administrator-reviewed SFT export. Its
consent, review checklist, manifest and restricted visibility are checked by the
existing contribution importer; the training configuration must specify
`objective: "sft"`. This does not grant public redistribution rights. Training
artifacts must remain in a location appropriate for restricted data.

The initialization selector distinguishes random initialization, exact resume
and a parent with a fresh optimizer. Provide a checkpoint only for the latter
two modes. An optional independent public validation dataset can be selected;
lineage contamination checks still apply. Parent training preserves architecture
and tokenizer, including an explicitly expanded vocabulary candidate. Run
configuration records the source type, parent/resume path and validation digest.

## Caché local de secuencias en disco

`python -m ai_home.foundation ... --sequence-cache-dir RUTA_NUEVA` evita crear
todas las secuencias rellenadas como listas Python y tensores residentes antes
de entrenar. Escribe filas int64 (entrada/etiqueta) de forma incremental y las
mapea con `torch.from_file`; el muestreo sigue cargando los lotes elegidos al
dispositivo. Aplica a train, validación y replay, con el mismo orden, padding,
desplazamiento causal y máscaras SFT. El valor predeterminado conserva la vía
en memoria. Cada ejecución, incluida una reanudación, exige un directorio nuevo.

Cada partición conserva un manifiesto con dimensiones, tamaño, endian y SHA256.
No se importan cachés existentes ni se reutilizan archivos no verificados.
Estas cachés son datos derivados locales: pueden contener conversaciones cuyo
entrenamiento fue autorizado y no deben publicarse como corpus públicos. No hay
cifrado adicional. Ocupan `filas * contexto * 16` bytes más manifiestos; errores
pueden dejar cachés parciales y no se borran automáticamente.

Límites: el corpus, el tokenizador y la tokenización de cada documento siguen en
RAM. El sistema operativo gestiona las páginas mapeadas; esto no garantiza un
límite duro de RSS ni elimina la necesidad de disco/RAM suficientes. No constituye
todavía ingesta de corpus grandes en streaming. La caché usa endian nativo y es
local, no un formato portátil de distribución.

Cuatro pruebas locales aprobadas: igualdad de secuencias causal/SFT y muestreo,
rechazo de directorios existentes, entrenamiento/reanudación idénticos bit a bit
frente a memoria tanto en CPU como en CUDA. Se conserva configuración del modelo,
optimizador y RNG del checkpoint; la ubicación de caché no cambia el experimento.

## Medición local de memoria de preparación

`python -m ai_home.foundation.sequence_benchmark --output-dir RUTA_NUEVA`
ejecuta cada vía en un proceso separado con dos documentos sintéticos propios
(2.240.000 bytes de texto), tokenizador sin merges y contexto 256. Ambas producen
8.752 filas con SHA256 idéntico de inputs y etiquetas. Se observa el pico de
working set del proceso en Windows y, en otras plataformas compatibles, max RSS.
Incluye Python/PyTorch; no equivale a memoria exclusiva del corpus.

| Medida local | En memoria | Caché |
| --- | --- | --- |
| Crecimiento del pico durante preparación | 70,6 MiB | 25,7 MiB |
| Crecimiento del pico tras lectura completa/hash | 73,6 MiB | 37,8 MiB |
| Tiempo de preparación | 0,262 s | 0,170 s |

La lectura completa posterior fuerza acceso a todas las filas, de modo que la
comparación no se limita a un mapeo todavía sin leer. La caché ocupa 35.848.192
bytes de datos más manifiesto. Informe: `SEQUENCE_MEMORY_BENCHMARK.json`.
Se hizo una primera medición y una segunda al añadir la observación tras lectura;
la tabla corresponde a la segunda, no a una selección por mejor resultado.

Es un microbenchmark de dos documentos en este equipo, con páginas/allocator
gestionados por el SO. No mide memoria total de entrenamiento, coste del BPE,
corpus diversos, discos lentos ni rendimiento distribuido. No garantiza esa
reducción en otros tamaños o máquinas ni sustituye límites de recursos.

### Cancelación durante preparación de secuencias

El evento de cancelación del entrenador se comprueba también antes de cada
documento, mensaje SFT y secuencia emitida, tanto en memoria como con caché.
Detener esta fase eleva `TrainingCancelled` y no crea un checkpoint. Los archivos
parciales se cierran y permanecen sin manifiesto de finalización; no se reutilizan
automáticamente y no se borran sin intervención. La CLI conserva su salida de
cancelación antes de actualización. No se interrumpe el interior de una llamada
individual a `tokenizer.encode`, ni el entrenamiento inicial del BPE; la latencia
de cancelación puede depender del documento actual.

Seis pruebas de caché aprobadas, incluyendo cancelación después de escribir una
fila, ausencia de checkpoint/manifiesto y liberación del archivo, además de
igualdad causal/SFT y reanudación CPU/CUDA.

La limitación histórica del BPE se ha reducido: el entrenador transmite ahora
cancelación a aprendizaje/codificación BPE y comprueba bloques durante conteo y
fusión. Ver TOKENIZER.md. Persisten operaciones nativas no interrumpibles; no se
promete una latencia máxima de parada.

### Caché desde el escritorio

El formulario «Entrenar Foundation» incluye «Guardar secuencias en disco para
reducir RAM», desmarcado por defecto. Al iniciar captura el valor en el hilo de
interfaz y lo entrega al trabajo local. Crea `sequence-cache` dentro de la nueva
carpeta de ejecución y registra `disk_sequences` en `configuration.json`.
Reanudar utiliza una carpeta/caché nueva; no reutiliza archivos previos.

Tk real comprobó presencia, estado inicial y conmutación. La prueba del trabajo
local ahora pausa y reanuda con caché y compara los pesos con una ejecución
completa en memoria. La disposición visual del formulario modificado queda
pendiente de inspección; esta comprobación Tk fue funcional con ventana oculta.

### Smoke del formulario con caché

`scripts/verify_desktop_training.py --output-dir NUEVO` crea un corpus propio y
configuraciones pequeñas, abre Tk real con ventana oculta, rellena el formulario,
activa la casilla y pulsa Entrenar. Usa el hilo y sondeo de eventos reales. Exige
cuatro pasos completos, checkpoint cargable, cambio de embeddings, manifiestos
y tamaños correctos de ambas cachés, botón rehabilitado y ruta final mostrada.
Solo aísla preferencias/memoria locales; no sustituye el entrenador ni sus pesos.

Ejecución CPU Windows aprobada: 6.672 parámetros, cambio L2 de embeddings
0.0358241, pérdida de validación 5.60127. No es una prueba de calidad del modelo.
Informe `DESKTOP_CACHED_TRAINING_SMOKE.json`; artefactos locales en
`artifacts/desktop-cached-training-v2`. CI incorpora el mismo smoke con wheel
instalado bajo Xvfb y retiene configuración/resultado. Su ejecución remota y la
revisión visual del formulario siguen pendientes.

## Separación explícita por familias documentales

Un registro puede declarar `split_group_sha256` (64 caracteres hexadecimales en
minúsculas), huella de una familia/fuente asignada por quien prepara el corpus.
Las variantes relacionadas deben compartirla. La validación rechaza formatos
inválidos y duplicados de texto con grupos contradictorios.

Cuando existe alguna etiqueta de grupo, la partición automática asigna cada
grupo por hash a train o validación, con umbral según `validation_fraction`.
Documentos sin etiqueta usan su hash de contenido como grupo individual. Bajo
este modo y una fracción fija, añadir documentos/grupos no reasigna los grupos
anteriores. El tamaño real de validación puede diferir del porcentaje objetivo.
Si queda vacía una partición, se exige ampliar grupos o aportar particiones
explícitas; no se rompe una familia para rellenarla. Activar este modo en un
corpus antes no agrupado puede cambiar su división: no es una migración automática
del holdout previo. Sin etiquetas se conserva la división histórica.

El entrenador rechaza grupos explícitos compartidos entre train/replay y
validación, y conserva `training_split_groups`/`validation_split_groups` en
checkpoints para impedir cruces futuros al continuar o reanudar. Checkpoints
antiguos sin estos campos solo tienen la protección de hashes documentales.
Estas etiquetas no prueban parentesco ni detectan duplicados semánticos: familias
omitidas, mal etiquetadas o renombradas pueden escapar al control. La revisión
del corpus y la evaluación independiente siguen siendo necesarias.

### Grupos emitidos por conectores

Las nuevas adquisiciones GitHub incluyen una familia basada en repositorio
(normalizado a minúsculas) y ruta (sensible a mayúsculas), sin el commit. Las
nuevas importaciones Wikipedia usan idioma y page_id, sin revisión ni URL del
dump. Así, versiones diferentes del mismo documento conservan el grupo aunque
cambie su contenido. La aprobación conserva el campo y su hash de metadatos lo
vincula a la revisión explícita. Los permisos siguen desactivados inicialmente.

Esto no agrupa automáticamente archivos renombrados, repositorios trasladados,
traducciones ni artículos relacionados. Los corpus/revisiones ya publicados no
se reescriben. Una adquisición nueva tiene metadatos nuevos y requiere su propia
revisión. Los índices conservan registros completos antes de fragmentarlos para
recuperación; los fragmentos de RAG no se convierten por ello en datos entrenados.

28 pruebas de conectores, revisión y grupos aprobadas: versiones GitHub,
revisiones Wikipedia, estabilidad y conservación después de aprobación. Se usó
HTTP simulado para el conector y dumps locales propios; no se descargó ni aprobó
contenido externo nuevo en esta validación.

### Grupos documentales en ampliaciones de vocabulario

La migración `expand_checkpoint` conserva los grupos de entrenamiento y
validación del padre y añade los grupos del corpus de ampliación a entrenamiento.
Rechaza documentos nuevos si pertenecen a una familia del holdout anterior;
proteger únicamente hashes de texto no bastaba. Se mantiene la comprobación de
hashes documentales existente. La prueba usa un checkpoint entrenado real,
rechaza una revisión distinta de la familia reservada y verifica el linaje de
un candidato ampliado. Checkpoints antiguos sin grupos conservan su limitación.

Pendiente: transportar y verificar este campo en el protocolo tipado de nodos
(distribuido), cuyo esquema actual todavía no admite la nueva etiqueta.

### Familias de contribuciones revisadas

El importador asigna `split_group_sha256` a cada contribucion raiz y a sus
correcciones, incluidas cadenas. La API conserva `root_contribution_id` en la
procedencia de nuevas correcciones, resuelta exclusivamente entre recursos del
propietario. El cliente no puede elegir esa raiz. El identificador se incluye
en el manifiesto congelado y su hash; los datos siguen restringidos a entrenamiento
revisado, sin permiso de distribucion a voluntarios.

Los exports antiguos sin raiz pueden importarse si contienen todos los antecesores
hasta la contribucion original. Si faltan o hay un ciclo, la importacion falla:
no se inventa una familia independiente. No se reescriben datasets congelados.
La nueva particion por familias puede dejar vacio train o validacion en corpus
pequenos, que requieren mas familias. Tambien cambia la huella preparada respecto
a importaciones anteriores; no se promete reanudacion exacta de checkpoints
creados antes de este cambio. El linaje antiguo sin grupos no permite comprobar
familias retrospectivamente. Copias aportadas como conversaciones independientes
no se reconocen semanticamente como una misma familia.

El trabajo local vincula la lectura inicial del export revisado con la del
entrenador mediante el hash de los registros preparados. Si el archivo cambia
entre ambas, incluso por otro export con manifiesto valido, falla antes de
preparar secuencias o actualizar pesos. Conserva `configuration.json` para
diagnostico, sin checkpoint ni resultado de exito. Se requiere un directorio
nuevo al reintentar. La lectura del export esta acotada sobre el mismo descriptor,
sin depender de una consulta previa del tamano. Esto verifica consistencia local;
no autentica la identidad del administrador ni sustituye su revision.

### Replay y comparacion con secuencias en disco

`python -m ai_home.foundation.continual --disk-sequences` extiende la opcion local
al entrenamiento incremental, replay, validacion y comparaciones padre/candidato.
El resto de argumentos del comando permanece obligatorio. La API Python admite
`run_candidate(..., disk_sequences=True)`. Cada conjunto se escribe en un
subdirectorio nuevo de `sequence-cache` dentro del experimento; el informe registra
la opcion. La preparacion de las comparaciones tambien atiende la cancelacion.

No cambia el muestreo, el criterio de regresion ni la aprobacion: el resultado
sigue siendo candidato y requiere evaluacion independiente. Los archivos pueden
contener texto privado tokenizado, ocupan disco y no se publican automaticamente.
No se garantiza un limite estricto de RAM ni se reutilizan caches existentes.
Pruebas CPU comparan exactamente metricas con/sin disco y pesos del candidato
contra una ejecucion reanudada en memoria, conservando linaje y restricciones.

La comparacion del ciclo continuo atiende cancelacion entre lotes de evaluacion
(y antes de devolver el resultado del ultimo lote). Una cancelacion conserva
el checkpoint candidato ya escrito, pero no produce `report.json` de comparacion.
Restaura el modo train/eval previo del modelo. El lote o kernel en curso termina;
no es una interrupcion instantanea. La iteracion usa vistas por lote sin crear
una tupla de vistas para todas las secuencias. La validacion periodica interna
del entrenador conserva su comportamiento de finalizar la validacion y guardar
el checkpoint al cancelar un entrenamiento.

Las comparaciones de perdida registran `checkpoint_sha256` de los pesos realmente
cargados por el lector seguro. El ciclo continuo contrasta esa identidad con el
padre usado para entrenar y con el candidato guardado; comprueba tambien los
archivos al terminar. Un cambio detectado aborta sin informe de comparacion.
El calculo adicional de hashes lee por bloques, sin materializar el checkpoint
completo en RAM. Esto no bloquea escrituras externas posteriores ni autentica
al autor de los pesos; los consumidores deben seguir verificando su hash.
