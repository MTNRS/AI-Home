# Cómputo Foundation en nodos

## Tokenización distribuida implementada

`foundation.tokenize.v1` utiliza el scheduler, autenticación, leases y NodeAgent
existentes. No ejecuta código del cliente. La API autenticada
`POST /api/v1/jobs/foundation/tokenize` recibe `payload` con `records` y
`tokenizer`, `idempotency_key` y `allow_public_node_processing: true` explícito.
Los registros usan el esquema de corpus público de TRAINING.md y el tokenizador
usa el formato propio ByteBPE. Nunca se conecta una exportación de conversaciones
restringidas a esta ruta.

Se aceptan 2–32 documentos, hasta 16.384 caracteres por documento y 64 KiB por
payload serializado. El BPE admite como máximo 250 merges, referencias válidas
y piezas de hasta 8.192 bytes. Se filtran patrones comunes de secretos y correos
en contenido y metadatos. La declaración pública y los derechos deben revisarse
por el operador; el filtro no demuestra anonimización completa ni licencia válida.

El nodo anuncia sus tipos de trabajo al registrarse. Clientes antiguos conservan
solo math.sum.v1; el cliente actualizado anuncia además tokenización. El scheduler
excluye tokenización si la RAM total declarada es inferior a 256 MiB, y el worker
comprueba RAM disponible antes de ejecutar. Es un límite conservador, no un
benchmark ni una reserva de memoria. Datos de hardware no confiables no prueban
rendimiento. No se requiere GPU para este trabajo.

Resultado: artefacto JSON versionado con ids de tokens por documento, hashes
de documentos/dataset/tokenizador, token_count y artifact_sha256. El servidor
recalcula el resultado completo antes de aceptarlo, bloqueando resultados
alterados. Esta validación duplica trabajo; no se reclama aceleración neta ni
un protocolo escalable de verificación remota. El artefacto no se consume aún
automáticamente en el trainer, que sigue tokenizando su entrada revisada.

## Demostración real del 2026-11-26

```powershell
$env:PYTHONPATH="services/api/src"
python -m ai_home.foundation.node_smoke --output-dir artifacts/foundation-node-smoke-v2
```

Usar un directorio nuevo. Se inicia un servidor limitado a loopback, una base
SQLite de prueba y dos procesos worker con credenciales distintas. Cada proceso
completa un shard mediante HTTP real. Se guardan tokenizador, artefactos, base
de prueba y reporte; las credenciales se pasan por stdin, no por argumentos ni
por el informe. Se apaga el servidor al terminar. Ambos nodos se ejecutan en un
solo ordenador y trabajan secuencialmente; no se demuestra una red entre equipos.

Resultado conservado: NODE_TOKENIZATION_SMOKE.json. Dos trabajos completados,
56 tokens y dos artefactos verificados. Los archivos locales están en
artifacts/foundation-node-smoke-v1, excluidos de Git.

## Pendiente

Validación de entrenamiento neuronal entre equipos distintos,
scheduling por GPU/VRAM/red/rendimiento, renovación de
leases para entrenamientos largos y prueba entre equipos. Tokenizar datos no
es entrenar pesos. PostgreSQL concurrente sigue pendiente de servidor de prueba;
SQLite no demuestra garantías de concurrencia de producción.

## Gradientes neuronales reales — 2026-11-26

Añadido `foundation.gradient.v1`, activado solo mediante
`AI_HOME_FOUNDATION_DISTRIBUTED_TRAINING_ENABLED=true`. Desactivarlo impide crear
y asignar nuevos trabajos de gradientes. Un lease ya emitido puede finalizar;
no se interrumpe remotamente un cálculo en curso.

El nodo debe anunciar ese tipo y `foundation_torch_version`, tener al menos
1 GiB de RAM total declarada y usar la misma versión de PyTorch que el
coordinador. NodeAgent requiere `allow_foundation_training=True`; los workers
de escritorio actuales no se inscriben automáticamente en trabajo neuronal.
El worker comprueba 1 GiB de RAM disponible. La tarea de referencia usa CPU FP32;
no promete compatibilidad de entrenamiento remoto con cualquier GPU.

Endpoint: `POST /api/v1/jobs/foundation/gradient`, con `idempotency_key`,
`allow_public_node_processing: true` y payload:

- `shard`: corpus público/tokenizador con el contrato de tokenización.
- `config`: FoundationConfig propia, máximo 2 capas, anchura 64, contexto 128,
  hidden 128 y 512 tokens; máximo 50.000 parámetros.
- `parameters`: tensores FP32 base64 con nombres y formas comprobados.
- `torch_version`: versión exacta del runtime.
- `weights_public_and_redistributable: true`: declaración explícita del operador.

No se acepta pickle, URL de código ni programas enviados por el cliente. Se
limita el shard a 4.096 bytes de texto, se validan forma/longitud/finitud antes
de cargar pesos y se conserva el límite HTTP de artefacto. Permisos y licencia
siguen requiriendo revisión del operador; la declaración no constituye una
verificación jurídica ni demuestra ausencia de datos memorizados en pesos.
No usar checkpoints derivados de conversaciones restringidas.

El nodo calcula pérdida causal y gradientes medios sobre su shard. El artefacto
liga gradientes, tokens y documentos al hash del modelo/tokenizador. El servidor
recalcula y compara cada resultado (tolerancias 1e-6/1e-5), dentro del lease,
antes de marcarlo completado. Esta verificación duplica trabajo neuronal: sirve
para validar el protocolo, no para ahorrar cómputo ni resistir cargas masivas.

El coordinador agrega gradientes ponderados por tokens, recorta la norma y
actualiza con AdamW. Rechaza bases antiguas, shards duplicados y documentos
solapados. Solo consumir resultados completados y verificados de la API;
el hash del artefacto por sí solo no autentica al productor. Una prueba compara
la actualización agregada con la obtenida sobre el conjunto completo.

### Experimento reproducible

```powershell
$env:PYTHONPATH="services/api/src"
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
python -m ai_home.foundation.distributed_smoke --output-dir artifacts/distributed-gradient-smoke-v3
```

Usar un directorio nuevo. Dos identidades de nodo, procesos worker distintos,
HTTP real en loopback y dos rondas; cada nodo procesa un shard por ronda.
Los workers se lanzan secuencialmente para esta prueba SQLite. No se reclama
paralelismo eficiente ni una demostración en dos máquinas. El modelo tiene
28.192 parámetros inicializados desde cero; validación reservada:
5,765336 antes y 5,602052 después. Cambio L2 de embeddings: 0,183900.
Cada ronda conserva checkpoint, estado AdamW, hashes y artefactos de gradientes.
Se recarga el checkpoint final y se genera texto neuronal todavía incoherente.
Informe: `DISTRIBUTED_GRADIENT_SMOKE.json`.

El script crea corpus y pesos propios públicos, servidor efímero local y base
SQLite; transmite credenciales a los workers por stdin y apaga el servidor al
terminar. Artefactos de la ejecución en artifacts/distributed-gradient-smoke-v2.

### Límites restantes

No hay renovación de leases largos, reanudación del coordinador entre rondas,
cobro de inferencia, distribución GPU, medidas de red, verificación remota
eficiente ni prueba multihost. El checkpoint puede cargarse para inferencia y
conserva el optimizador, pero el CLI de entrenamiento local no lo trata como
resume de su propio scheduler. Continúan pendientes las garantías concurrentes
de PostgreSQL y el endurecimiento de producción.

## Recompensas financiadas integradas

Tokenización y gradientes aceptan reward_credits. Se reservan fondos del
solicitante al crear el trabajo y se abonan al propietario del nodo solo después
de verificar el resultado. La demostración --reward-credits 5 completó cuatro
trabajos neuronales y liquidó 10 créditos a cada worker. Las menciones anteriores
a créditos pendientes corresponden al estado previo. Detalles de atomicidad,
reembolso, migración y límites en CREDITS.md.


## CPU/CUDA volunteer gradient execution

`NodeAgent(..., allow_foundation_training=True, foundation_device="cuda")` and
`execute_job(..., foundation_device="cuda")` explicitly opt into CUDA execution
of the same bounded FP32 gradient task. CPU remains the default. CUDA requests
fail if unavailable or the estimated device memory does not fit; there is no
silent fallback. Public-data/weights restrictions and the 50,000-parameter task
budget remain. The wire format stays device-independent. The coordinator still
recomputes on CPU and uses the existing loss/gradient tolerances (unchanged).

Reproduce the mixed-device demonstration with the CUDA Python environment:

```powershell
$env:PYTHONPATH='services/api/src'
.venv-cuda\Scripts\python.exe -m ai_home.foundation.distributed_smoke --output-dir artifacts/distributed-mixed-new --rounds 2 --reward-credits 5 --worker-devices cpu cuda
```

`DISTRIBUTED_MIXED_CUDA.json` records two logical nodes over loopback HTTP,
separate worker processes, two aggregate updates, embedding change 0.1838997
and 10 verified internal test credits earned by each worker. The 28,192-parameter
checkpoint was reloaded and generated incoherent text. A CUDA test also observes
actual CUDA input tensors during worker forward calls, then CPU tensors during
independent verification, and aggregates the result with a CPU shard.

This is single-host heterogeneous execution, not multiple physical hosts or
multiple GPUs. Device choice is worker-local; scheduler matching still uses
supported task, RAM and PyTorch version. GPU throughput/VRAM/network-aware task
ranking, larger shards, multi-host efficiency and independent scalable validation
remain outstanding. Tiny CPU recomputation is intentionally expensive evidence,
not a proposed production scaling strategy.

### Recoverable aggregation boundary

`python -m ai_home.foundation.distributed_round PARENT.pt ENVELOPES.json NEW.pt`
reconstructs a next round from an existing public distributed checkpoint, including
AdamW state. Envelopes contain the typed `payload` and `result` for each shard;
2–8 shards are accepted and the CLI bounds their file to 16 MB. Every gradient is
independently recomputed on CPU before aggregation. Overlap with the parent's
fixed validation hashes is refused. The new candidate drops inherited validation
scores, records parent/request/dataset identities and publishes complete bytes
exclusively. Existing targets are reusable only for the identical parent and
ordered shard request; other requests fail. A retry never mutates the parent or
adds another optimizer update to an already committed output.

The HTTP smoke now persists round envelopes before its in-memory update. This
allows explicit reconstruction after a coordinator interruption from round 1
onward; initial round bootstrap remains the smoke's responsibility. Example:

```powershell
python -m ai_home.foundation.distributed_smoke --output-dir artifacts/recovery --rounds 2 --worker-devices cpu cuda
python -m ai_home.foundation.distributed_round artifacts/recovery/round-1.pt artifacts/recovery/round-2-envelopes.json artifacts/recovery/recovered-round-2.pt
```

Actual mixed CPU/CUDA loopback execution reconstructed identical weights to the
uninterrupted second round, and retry preserved the committed bytes. Evidence is
in DISTRIBUTED_RECOVERY.json. Tests inject interruption after aggregation but
before publication and compare recovery with uninterrupted AdamW exactly.
This covers the aggregation commit boundary only: automatic service restart,
persisted job discovery/resubmission, multi-host failover and credit/job recovery
coordination remain outstanding. Local checkpoint/envelope storage is trusted;
this is not an authenticated remote coordinator protocol.

### Restartable job submission and discovery

`foundation.coordinator` adds an immutable round plan binding the server URL,
owner account ID, exact parent checkpoint, public typed payloads and per-job reward.
Plan creation requires explicit public-node consent and checks parameter/tokenizer
identity and disjoint shards/validation data before any jobs are submitted. Plans
contain public training material but no login token. Keep them in local trusted
storage; a hash detects accidental changes, not a malicious operator rewriting it.

```powershell
python -m ai_home.foundation.coordinator plan round-1.pt payloads.json round-plan.json --server https://your-server.example --owner YOUR_USER_ID --allow-public-node-processing --reward-credits 3
# Supply the current owner's session token via AI_HOME_COORDINATOR_TOKEN.
python -m ai_home.foundation.coordinator advance round-plan.json round-1.pt round-2.pt --server https://your-server.example
```

`payloads.json` is a list of the existing typed gradient payloads (the `payload`
fields in smoke envelopes). `advance` verifies server/account/parent identity,
then submits the same deterministic owner-scoped key for each shard. Existing jobs
are returned with their current status, including after a lost creation response.
A pass returns `waiting` with IDs/statuses or verifies and commits the completed
round. Re-run the same command after a process restart; a fresh access token is
allowed for the same account. Failed/cancelled jobs stop the round instead of
creating replacement keys or spending another reward. Already funded jobs remain
subject to the existing server expiry/cancel/refund rules.

This is an explicit one-pass CLI/API coordinator, not a background daemon. It does
not persist tokens, poll forever, restart workers, or cancel outstanding jobs on
CLI exit. The operator can inspect/cancel those jobs through existing endpoints.
It does not bootstrap the first distributed checkpoint or provide multi-host
failover. Independent verification remains intentionally expensive and bounded.

An API integration test loses the second job reply after its database commit,
restarts submission, verifies both original IDs and a single six-credit reserve,
executes real gradient workers, commits the next checkpoint, and retries without
changing bytes or job count. It also rejects a different authenticated owner.
This test uses the real FastAPI/database/ledger implementation in-process; the
separate prior recovery smoke provides real loopback CPU/CUDA transport evidence.

### Real coordinator process restart smoke

Run the existing HTTP experiment with `--restart-coordinator` and at least two
rounds to exercise four independent coordinator CLI processes for round 2:

```powershell
python -m ai_home.foundation.distributed_smoke --output-dir artifacts/coordinator-restart --rounds 2 --reward-credits 3 --worker-devices cpu cuda --restart-coordinator
```

Round 1 bootstraps the parent checkpoint. The first coordinator process creates
round-2 jobs and exits waiting; the second reconstructs the same request from the
plan and finds identical IDs with unchanged reserves. CPU/CUDA worker processes
complete those jobs over HTTP. A third coordinator process commits the new round,
and a fourth verifies the already committed result. The smoke compares recovered
weights exactly with an uninterrupted reference update, checks unchanged output
bytes on retry, and asserts exactly two jobs per round. Tokens are passed through
child-process environment only and are not stored in the plan/report.

The actual two-round run in COORDINATOR_RESTART_SMOKE.json completed all four
coordinator processes, paid six test credits to each worker and left no reserved
credits. It used one physical Windows host and loopback networking. This proves
fresh-process reconstruction at those boundaries; it does not demonstrate forced
process termination during every instruction, API-server restart, automatic
service supervision or multi-host failover. The separate fault-injection test
covers interruption before checkpoint publication and lost HTTP creation replies.

### Bounded coordinator supervision

Add `--wait-seconds 300 --poll-seconds 5` to `coordinator advance` to observe the
same round until committed or until the polling deadline. Wait is bounded to
1–3600 seconds; polling interval is 1–60 seconds. The default remains a single
pass. A transport failure retries the identical durable job keys, including after
a lost creation response. HTTP status failures (authentication, permission,
rate limits or server errors), invalid plans and failed/cancelled jobs remain
visible errors rather than being silently retried.

The API `supervise(..., cancel=event)` supports cooperative stopping between
passes. CLI Ctrl+C returns `interrupted`; timeout returns `timed_out` with the last
observation and attempt/transport-error counts. Neither action cancels funded
server jobs or assumes that no checkpoint was committed. Re-run the same plan to
reconcile after an interruption; use the explicit server cancellation API to
cancel jobs. HTTP calls and independent gradient verification already in progress
can extend beyond the polling deadline; this is not a hard wall-clock kill timer.
The helper is not a Windows service or multihost failover manager.

Control-flow tests use a virtual clock to verify retry, deadline, stop and refusal
to retry HTTP policy errors. The real API/ledger/gradient test additionally uses
supervision to reconcile an already committed checkpoint without changing bytes.
These tests are distinct from the real HTTP fresh-process smoke described above.

### Measured scheduling from the coordinator CLI

Optional per-job performance/service constraints can now be supplied with
`coordinator plan ... --scheduling requirements.json`. See
[constraints, bootstrap requirements and limits](MEASURED_SCHEDULING.md).
The file is bounded to 64 KiB, constraints are part of the immutable plan hash,
and retry sends the same requirements. Planning remains local and does not
reserve credits; `advance` performs authenticated idempotent submission.

## Participación Foundation desde escritorio

DISTRIBUTED COMPUTE permite habilitar expresamente entrenamiento Foundation
experimental y elegir CPU o CUDA. Ambos controles se capturan al iniciar el
nodo; entrenamiento está desactivado por defecto y no se persiste. CUDA no
retrocede automáticamente a CPU. Se mantienen los límites de memoria, tamaño,
procedencia pública y verificación independiente del protocolo de gradientes.

Al iniciar, el escritorio llama a `PUT /api/v1/node/capabilities` autenticado con
la credencial del propio nodo. Actualiza los tipos permitidos y la versión real
de PyTorch; al desactivar elimina la capacidad de gradientes y su versión.
El servidor conserva su interruptor global de autorización. La actualización
no admite seleccionar otro nodo y descarta la muestra anterior de rendimiento
computacional. Los trabajos con filtros de rendimiento requieren una nueva
muestra obtenida tras trabajo compatible sin esos filtros.

Se requiere cliente y servidor actualizados; un servidor sin esta ruta produce
un error real y el escritorio no inicia reclamaciones. Cambiar controles requiere
detener y reiniciar el nodo. Detener espera al trabajo en curso; no revoca una
asignación ya entregada. Esta integración no prueba entrenamiento multihost ni
calidad lingüística generalista.

Pruebas locales: 37 aprobadas de sondas/nodos/gradientes antes de ampliar la
comprobación de opt-out; después, 42 de gradientes/escritorio/privacidad,
incluida desactivación/reactivación efectiva en el scheduler. Tk real verificó
los controles por defecto y su conmutación. Revisión visual pendiente.

## Smoke del nodo desde Tk real

`scripts/verify_desktop_node.py --output-dir NUEVO --device cpu` crea una API
loopback con SQLite migrada, dos cuentas sintéticas y tres créditos de prueba.
Usa la ventana Tk real, `_start_contribution`, detección de hardware, registro
HTTP, actualización de capacidades, sonda opcional y bucle real del nodo.
Un gradiente calculado con pesos propios inicializados desde cero debe superar
la verificación del servidor y liquidar los tres créditos al trabajador.

Solo se sustituyen la persistencia local de preferencias/memoria y keyring por
almacenamiento efímero; no se modifican credenciales reales. Los errores de la
interfaz hacen fallar el smoke sin dejar un diálogo modal esperando interacción.
El script conserva payload, resultado, base de datos y reporte, sin publicar
credenciales en el reporte. Es una verificación funcional con ventana oculta,
no una revisión visual ni una demostración de entrenamiento en varios hosts.

Ejecutado correctamente en Windows con CPU y RTX 4070 Super CUDA. Informes:
`DESKTOP_NODE_CPU_SMOKE.json` y `DESKTOP_NODE_CUDA_SMOKE.json`. En ambos casos:
trabajo completado, trabajador gana 3 créditos, solicitante disponible/reservado
0, capacidades y dispositivo reportados correctos, sonda publicada. Una primera
revisión del arnés falló por importar messagebox desde el módulo equivocado;
se corrigió y las dos ejecuciones finales pasaron. CI incorpora ahora el mismo
flujo CPU contra el paquete instalado bajo Xvfb; su ejecución queda pendiente.

## Aislamiento del nodo al cambiar de cuenta

Cambiar de perfil solicita detener el bucle del nodo, invalida sus mensajes de
estado/error pendientes y reinicia los consentimientos de contribución,
entrenamiento y sondeo. Un trabajo ya iniciado puede terminar y liquidarse a
la cuenta original; no se transfiere a la nueva cuenta. Antes de reutilizar una
credencial guardada, el escritorio consulta los nodos activos de la cuenta
actual en el servidor y exige encontrar el identificador guardado. Si no puede
verificarlo, no inicia el nodo. Esta consulta usa el listado existente limitado
a los 100 nodos más recientes: un nodo antiguo fuera de esa lista tampoco se
reutiliza. No borra ni revoca automáticamente credenciales de otra cuenta.

Verificado con Tk y HTTP reales en `verify_desktop_node.py`: una cuenta crea el
nodo, completa un gradiente y gana créditos; al cambiar a la otra cuenta, el
intento de reutilizarlo se rechaza y no nace otro trabajador. Los estados
pendientes anteriores no sustituyen el estado del nuevo perfil. Informe:
`DESKTOP_NODE_ISOLATION_SMOKE.json`. La prueba de cambio de perfil visual comprueba
también la señal de parada y los consentimientos; 36 pruebas de escritorio y
privacidad aprobadas. No se simula cálculo neuronal en este ensayo.

### Consulta directa de propiedad del nodo

La comprobación previa al arranque usa ahora `GET /api/v1/nodes/{node_id}` y ya
no depende del listado de 100 nodos. La consulta exige sesión de usuario y filtra
por propietario y ausencia de revocación; un nodo ajeno, inexistente o revocado
produce el mismo 404. No expone la credencial ni su hash. El listado conserva su
límite anterior. Cliente y servidor deben actualizarse juntos.

Validación local: 25 pruebas de nodos/trabajos aprobadas y nueva ejecución del
smoke completo CPU/Tk/HTTP con bloqueo de reutilización entre cuentas
(`artifacts/desktop-node-lookup-v1`). CI Foundation 45839308567 de 28fc6c8 terminó
correctamente, incluido el nodo de escritorio instalado en Linux bajo Xvfb.

## Familias documentales en tareas distribuidas

Los registros públicos tipados admiten `split_group_sha256` opcional y validado.
Cuando falta se omite al serializar, conservando la representación de registros
anteriores. Los artefactos de tokenización conservan la etiqueta cuando existe.
Planificador y nodos deben estar actualizados para admitir trabajos etiquetados;
no se elimina el campo para hacerlos pasar por un trabajador antiguo.

`create_plan` rechaza familias presentes en `validation_split_groups` del padre
antes de publicar trabajos. `commit_round` repite esa comprobación antes de
agregar/publicar y conserva la unión de familias de entrenamiento. Los registros
sin etiqueta y padres antiguos solo disponen de la información de linaje que
aportan. La API genérica de trabajos no conoce el holdout de un checkpoint:
estas comprobaciones pertenecen al plan y al commit de la ronda. No constituyen
una certificación independiente de que las etiquetas de fuente sean correctas.

Pruebas locales: 18 de rondas/coordinador/nodos/gradientes aprobadas; después,
ocho de nodos tras añadir conservación de grupos en tokenización y compatibilidad
de serialización sin etiqueta (conjunto solapado). Los rechazos y una ronda
permitida se verificaron con cálculo real de gradientes y checkpoint resultante.

### Evaluacion distribuida experimental de perdida

`foundation.distributed_evaluation.execute_evaluation` calcula perdida causal
sin backward ni optimizador para un payload publico con los mismos limites de
arquitectura, tensores, runtime y datos que las tareas de gradientes. Devuelve
conteo de tokens, perdida media, hashes de documentos/dataset y huella de pesos.
No usa pickle ni ejecuta codigo del remitente. `verified_evaluation_result`
recomputa en CPU y compara identidad y perdida (rtol 1e-5, atol 1e-6).

Probado en CPU y CUDA con pesos propios: perdida concordante con el recorrido de
gradientes, entrada y RNG CPU intactos, resultados alterados rechazados y
restricciones de datos/pesos preservadas. La reserva de memoria es conservadora:
reutiliza el presupuesto del entrenamiento de referencia.

El ejecutor esta conectado a `POST /api/v1/jobs/foundation/evaluate`, con tipo
`foundation.evaluate.v1`. El servidor exige
`AI_HOME_FOUNDATION_DISTRIBUTED_EVALUATION_ENABLED=true` (por defecto false),
independientemente del permiso de entrenamiento. La peticion autenticada contiene
`payload` (mismo esquema acotado de pesos y shard que gradientes),
`allow_public_node_processing: true`, `idempotency_key` y `reward_credits` opcional.
No admite restricciones de scheduling basadas en mediciones de gradientes.

El nodo debe anunciar esa capacidad y la version exacta de PyTorch. El scheduler
exige RAM suficiente, respeta snapshots de recursos/caducidad y VRAM CUDA, y no
entrega estas tareas a trabajadores antiguos. En Python se habilita mediante
`NodeAgent(..., allow_foundation_evaluation=True)`, seguido de
`configure_capabilities()` y el bucle habitual `run_once()`. El permiso es
independiente de `allow_foundation_training`. El panel de nodos del escritorio
incluye una casilla de evaluacion separada, desactivada por defecto; usa el
dispositivo Foundation seleccionado. Cambiar de perfil revoca ambos permisos
locales y detiene el bucle anterior. No se habilita por instalar una actualizacion.

Resultados alterados se rechazan antes de liquidar creditos. Se reutilizan reservas,
leases, idempotencia y liquidacion transaccional. El test de API usa transporte
HTTP en proceso con FastAPI y NodeAgent reales, dos trabajos remunerados y
recomputacion CPU. El smoke por sockets se describe debajo; siguen sin verificarse
dos equipos fisicos.
El endpoint de bajo nivel no aporta linaje de entrenamiento. Para preparar una
evaluacion contra un checkpoint debe usarse el coordinador descrito debajo;
la perdida no demuestra por si sola capacidad linguistica.

### Coordinador de evaluacion publica

`foundation.evaluation_coordinator.create_plan(checkpoint, shards, output,
server=..., owner_user_id=..., reward_credits=0,
allow_public_node_processing=True, weights_public_and_redistributable=True)`
crea un plan inmutable con uno a ocho fragmentos publicos. Cada fragmento mantiene
los limites del ejecutor. Exige checkpoint publico con linaje, rechaza textos de
entrenamiento y familias entrenadas cuando estan etiquetadas, y evita documentos
duplicados entre fragmentos. Grupos etiquetados requieren linaje de grupos en el
checkpoint. Los consentimientos de datos y redistribucion de pesos son separados.

`advance(client_httpx_autenticado, plan_path, checkpoint, output_report)` comprueba
servidor, propietario, identidad del checkpoint y payloads; somete trabajos con
claves idempotentes y retorna `state=waiting` hasta completarlos. Repetir la llamada
tras reiniciar el observador conserva trabajos y reservas. Fallos/cancelaciones
no crean reemplazos automaticamente. El resultado se recomputa en CPU tambien
en el coordinador y se agrega por tokens, conserva hashes y jobs y siempre
indica `promoted=false`. La publicacion es exclusiva; repetir un informe identico
es valido, sustituir uno diferente produce error. No se guardan credenciales.

La prueba HTTP en proceso compara dos fragmentos con evaluacion monolitica real,
verifica reanudacion, seis creditos totales sin duplicacion y rechazo de otra
cuenta. No demuestra multiples equipos fisicos. Los registros/linaje declarados
no prueban parentesco semantico ni autoria. Un holdout publico puede conocerse;
no se presenta como benchmark secreto o evaluacion independiente de capacidades.

El coordinador dispone de CLI reproducible:

```powershell
python -m ai_home.foundation.evaluation_coordinator plan --checkpoint parent.pt --shards shards.json --owner USER_ID --server http://127.0.0.1:8000 --output evaluation-plan.json --reward-credits 3 --allow-public-node-processing --weights-public-and-redistributable
python -m ai_home.foundation.evaluation_coordinator advance --checkpoint parent.pt --plan evaluation-plan.json --server http://127.0.0.1:8000 --output evaluation-report.json
```

`shards.json` es un array de listas de registros publicos, limitado a 1 MB en
esta entrada. `advance` obtiene la credencial de `AI_HOME_COORDINATOR_TOKEN`,
no de argumentos ni del plan; requiere la cuenta propietaria. Cada invocacion
hace una pasada y devuelve `waiting` o `evaluated`. Repetirla tras ejecutar los
trabajadores o tras reiniciar reanuda los mismos trabajos. Ctrl+C detiene el
observador, no cancela reservas o tareas ya enviadas. No sigue redirecciones
HTTP ni utiliza proxies ambientales. La interfaz grafica sigue pendiente.

### Smoke de evaluacion con sockets y procesos reales

```powershell
python -m ai_home.foundation.evaluation_smoke --output-dir artifacts/evaluation-run --worker-devices cpu cuda
```

Crea una base SQLite aislada con migraciones, checkpoint propio entrenado, corpus
sintetico autorizado, dos cuentas de trabajadores y un servidor HTTP en loopback.
Cada trabajador ejecuta una tarea en su propio proceso. Tres procesos CLI sucesivos
del coordinador someten, agregan y repiten el informe sin duplicar trabajos o
creditos. Exige carpeta nueva; no utiliza cuentas ni datos personales existentes.
Credenciales efimeras se transmiten por entorno, no se guardan en informes.

Ejecucion local CPU/CUDA: 468 tokens, perdida distribuida 5.634302514231103 frente
a 5.634302383814102 local; tres creditos por trabajador, dos jobs completados,
checkpoint intacto. [Informe](DISTRIBUTED_EVALUATION_SMOKE.json). Son medidas de
servicio en un equipo, no ancho de banda fisico ni escalabilidad multihost. No
miden calidad linguistica. CI incorpora el recorrido CPU/CPU desde el paquete
instalado y publica su informe; confirmar su ejecucion remota antes de afirmarla.

### Evaluacion desde el nodo de escritorio

`scripts/verify_desktop_node.py --output-dir CARPETA_NUEVA --device cuda --task evaluate`
verifica con Tk real el registro, capacidades, bucle del trabajador, perdida real,
verificacion del servidor y tres creditos. El permiso de entrenamiento queda
apagado y la capacidad de gradientes no se anuncia. Al cambiar de cuenta se
borran ambos permisos, se descartan estados anteriores y se bloquea reutilizar
el nodo de otra cuenta. Solo persistencia local/keyring se aislan en memoria.

Ejecutado localmente con CUDA y CPU; repetido el recorrido previo de gradientes
CPU para comprobar que conserva su funcionamiento. Informes:
[CUDA](DESKTOP_EVALUATION_CUDA_SMOKE.json) y [CPU](DESKTOP_EVALUATION_CPU_SMOKE.json).
Las ventanas del smoke permanecen ocultas: demuestra ejecucion funcional, no
legibilidad ni distribucion visual. CI incorpora la variante instalada CPU;
la confirmacion remota de ese nuevo paso queda pendiente.

El CLI de evaluacion admite `advance --wait-seconds 300 --poll-seconds 5` para
seguir el plan durante un intervalo acotado. Limites: espera 1-3600 s y sondeo
1-60 s; sin `--wait-seconds` conserva la pasada unica. La API Python ofrece
`supervise(..., wait_seconds=..., poll_seconds=..., cancel=evento)`.

El observador compartido con el coordinador de gradientes reintenta errores de
transporte usando las mismas claves idempotentes. Errores HTTP de autenticacion,
politica o estado se propagan. Devuelve `evaluated`, `timed_out` o `stopped`,
contador de intentos y ultima observacion cuando procede. Parar o agotar la espera
no cancela trabajos ni libera sus reservas. Una peticion HTTP en curso conserva
su timeout propio, de modo que el plazo no es un limite duro de tiempo total.

El smoke tambien inicia el observador CLI antes de lanzar los trabajadores, con
`--wait-seconds 60 --poll-seconds 1`. La ejecucion CPU/CUDA observada hizo nueve
pasadas, sin errores de transporte, antes de retornar el informe; el tercer
proceso repitio la recogida sin duplicar jobs/creditos. [Informe de observacion](DISTRIBUTED_EVALUATION_OBSERVER_SMOKE.json).
Si el smoke falla, cierra su proceso observador pendiente antes de desmontar el
servidor. Los contadores son evidencia de esa ejecucion, no un benchmark de red.


### Plan de evaluacion desde un dataset

El coordinador admite un dataset Knowledge publico como alternativa al JSON
manual de fragmentos. Debe contener solo el holdout autorizado; no se extraen
ni descartan silenciosamente documentos entrenados. Ejemplo en PowerShell:

```powershell
python -m ai_home.foundation.evaluation_coordinator plan `
  --checkpoint artifacts/model.pt --dataset artifacts/holdout.json `
  --shard-count 2 --owner OWNER_ID --server http://127.0.0.1:8000 `
  --output artifacts/evaluation-plan.json --reward-credits 3 `
  --allow-public-node-processing --weights-public-and-redistributable
```

La particion es determinista por hash de documento, independiente del orden
del archivo. Distribuye todos los documentos normalizados y deduplicados en
1 a 8 fragmentos (2 a 32 documentos por fragmento). No equilibra tiempos de
GPU ni cambia el texto para ajustarlo: si se exceden los limites del protocolo,
la creacion del plan falla. La lectura del dataset tiene presupuesto de 1 MB.
Se conservan licencia, permisos y linaje de grupos; el plan rechaza coincidencias
con documentos o familias entrenadas. Las familias pueden aparecer en varios
fragmentos del mismo holdout; esto no crea particiones train/test nuevas.

`--dataset` exige `--shard-count`; `--shards` mantiene el formato manual anterior
y no admite esa opcion. Crear el plan es local y no reserva creditos; `advance`
conserva la autenticacion, envio y verificacion descritos anteriormente.

El smoke `python -m ai_home.foundation.evaluation_smoke --output-dir NEW_DIR
--worker-devices cpu cuda` ahora recorre esta entrada mediante un proceso CLI
real. Conserva holdout.json versionado, plan, checkpoint y resultados; comprueba
que planificar no crea trabajos ni reservas, y que todos los documentos del
holdout llegan al plan. Inicia cuatro procesos del coordinador: planificacion,
primer envio, observacion y consulta idempotente final. El informe
DISTRIBUTED_EVALUATION_DATASET_SMOKE.json registra la ejecucion CPU/CUDA local:
468 tokens, dos trabajos y tres creditos liquidados por nodo. La CI reutiliza
el mismo smoke con CPU/CPU desde el paquete instalado; esta revision no se
declara validada remotamente hasta que termine su workflow.

Antes de cargar el checkpoint, el coordinador valida los fragmentos publicos,
la procedencia, los duplicados entre fragmentos, el numero de documentos y
el limite de 4096 bytes UTF-8 de texto por fragmento. Los errores de tamano
y duplicacion identifican el fragmento (numeracion desde uno). El texto no
se recorta para hacerlo caber. Las comprobaciones del linaje y de los tensores
siguen realizandose al cargar el modelo y validar los payloads completos.

### Planificacion desde escritorio

EVALUATION incluye Planificar evaluacion distribuida. El formulario pide
checkpoint, dataset holdout, archivo nuevo, servidor, ID del propietario,
fragmentos y creditos por fragmento. Ambos consentimientos empiezan desactivados
y son obligatorios. La validacion y escritura se ejecutan fuera del hilo Tk
mediante el coordinador existente; no se envian trabajos ni se reservan creditos.
El archivo contiene textos y pesos publicos. La ejecucion posterior sigue
disponible mediante advance autenticado; el formulario no sustituye ese paso.

La ventana pertenece al perfil y se cierra al cambiarlo; los resultados tardios
se descartan de la interfaz. Cerrar no cancela una escritura local iniciada.
Prueba Tk real: scripts/verify_desktop_evaluation_plan.py crea un plan con un
checkpoint entrenado y rechaza cero o un consentimiento. Integrada en la CI
del paquete instalado bajo Xvfb; no equivale a revision visual de DPI/layout.

El smoke Tk tambien fuerza una finalizacion posterior al cambio real de perfil:
comprueba descarte de exito/error tardios y reinicio de ambos consentimientos.
Mantiene el token sin modificar para verificar la guardia de ventana por si
misma. El caso exitoso conserva el archivo local producido tras el cierre;
no se presenta ese resultado en el nuevo perfil.
