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
