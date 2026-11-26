# Estado de implementación

Última actualización: 2026-11-23

## Fase actual

**Fases 3, 5 y 8 — Nodo voluntario, cliente BYOK y captura consentida
experimental; producción pendiente.**

## Completado

- Repositorio Git inicial conectado a GitHub.
- Principios, límites arquitectónicos y hoja de ruta documentados.
- Definido el enfoque inicial de monolito modular, PostgreSQL autoritativo y
  nodos voluntarios no confiables.
- Implementado un núcleo de libro mayor con asientos balanceados, saldos no
  negativos por cuenta, operaciones idempotentes y reserva/liquidación/
  reembolso componibles.
- Licencia Apache-2.0 añadida al código del repositorio.
- Backend inicial con configuración por entorno, API FastAPI, registro,
  login/logout con sesiones opacas revocables y endpoint de perfil propio.
- Integración opcional con Supabase Auth: registro/login mediante Auth REST,
  verificación de JWT con JWKS del proyecto, revocación local de sesiones y
  proyección local del UUID/correo sin guardar contraseñas. La clave secreta se
  limita a eliminación administrativa de cuentas.
- Migración `0005` permite perfiles con credenciales gestionadas externamente;
  el modo local de desarrollo conserva Argon2id y los tokens se almacenan como
  hashes.
- Migración Alembic inicial para identidad, sesiones y ledger, más Compose de
  desarrollo para PostgreSQL y CI Python 3.11/3.12.
- Pruebas de API para registro, normalización, autenticación, sesión, cierre y
  respuestas sin secretos.
- Exportación básica y eliminación de cuenta con confirmación de contraseña;
  la eliminación invalida sesiones y elimina nodos/trabajos del usuario.
- Endpoint `/health/ready` comprueba conectividad a la base de datos.
- Protocolo inicial de nodos con credencial aleatoria hasheada, informe de
  hardware por GPU y leases persistentes para el ejecutor determinista
  `math.sum.v1`.
- Los trabajos distribuidos requieren consentimiento explícito registrado; el
  scheduler valida el resultado, reintenta leases vencidos y limita intentos.
- Cliente de nodo Python ejecuta solo handlers tipados y permite una iteración
  de polling, sin ejecutar código arbitrario.
- Primera aplicación de escritorio Tk con cuenta, chat y aportación de nodos
  claramente opcional; tokens de nodo en el almacén seguro del sistema
  operativo.
- Adaptadores directos de escritorio para OpenAI, Anthropic, Google Gemini y
  endpoints OpenAI-compatibles. Claves guardadas en keyring por proveedor y
  endpoint; no se incluyen en preferencias ni se envían a AI@Home. Endpoints
  oficiales fijos y endpoint compatible remoto sujeto a HTTPS.
- UI BYOK advierte del procesamiento por terceros y permite guardar/eliminar
  claves con manejo explícito de errores del almacén seguro, sin fallback a
  texto plano.
- Migración `0003` y API autenticada para candidaturas de conversaciones:
  consentimiento/versionado/procedencia, opt-in doble apagado por defecto,
  filtro de secretos comunes, aislamiento por cuenta, consulta propia,
  retirada física antes de congelación y eliminación al borrar la cuenta.
- Migración `0004` y API de revisión protegida por secreto administrativo:
  aprobación exige checklist de términos/datos sensibles/derechos; rechazos
  eliminan el contenido. Candidaturas elegibles se pueden congelar en datasets
  versionados de hasta 100 muestras con snapshot y SHA-256 reproducible.
  Retiradas se sincronizan con el lock de fila; el snapshot congelado persiste
  al borrar la cuenta y no es reversible.
- La interfaz permite enviar una conversación solo tras opt-in y revisión
  expresa; permite consultar o retirar candidaturas propias. La revisión de
  contenidos se opera solo mediante API administrativa; no hay UI de revisión.
  Los datasets congelados no se consumen aún en entrenamiento.
- Proxy de chat opcional y autenticado en API para un proveedor
  OpenAI-compatible configurado por la instancia; clave solo del lado servidor.
- Inventario local verificado: 16 hilos lógicos, 31 GiB de RAM y una NVIDIA
  GeForce RTX 4070 SUPER con 12 GiB de VRAM; la detección usa `nvidia-smi`.
- CI aprobada previamente en Python 3.11/3.12 con PostgreSQL 17 para la línea
  base; los cambios BYOK/candidaturas aún requieren ejecución en CI/PostgreSQL.
- Suite local más reciente antes de este cambio: `python -m pytest -q` — 103
  pruebas aprobadas y 4 pruebas PostgreSQL omitidas por no haber servidor
  disponible localmente.
- Suite local tras implementar el pipeline del LLM: `python -m pytest -q` —
  120 aprobadas y 4 omitidas por no haber servidor PostgreSQL local.
- PyTorch no estaba instalado ni había corpus local al iniciar este cambio. Se
  añadió como extra opcional; la validación usó una versión CPU, no una sesión
  de entrenamiento real en GPU. La RTX 4070 SUPER de 12 GiB permite investigar
  un modelo pequeño, no asumir que se puede entrenar aquí un LLM competitivo.
- Implementado un GPT byte-level pequeño con pesos inicializados aleatoriamente,
  entrenamiento local con PyTorch opcional y carga de checkpoints locales. La
  app lo selecciona por defecto y muestra un error explícito hasta que exista
  un checkpoint; no hace fallback silencioso a proveedores externos.
- Importadores experimentales para Wikipedia en español y para conversaciones
  revisadas en español de OASST1 (Apache-2.0). Cada corpus lleva manifiesto,
  hash al entrenar y registros de atribución; el entrenamiento puede mezclar
  corpus con muestreo equilibrado. No se descarga contenido automáticamente.
- El código de entrenamiento y los importadores están implementados. Se
  verificó la importación real de OASST1 (3.054 ejemplos conversacionales en
  español tras filtros), pero no se entrenó/evaluó un checkpoint con ese corpus.
  El modelo byte-level es una prueba de concepto; no se afirma que pueda
  mantener conversaciones coherentes.
- Las pruebas incluyen adaptadores de proveedores con transportes simulados,
  keyring aislado, consentimiento, bloqueo de secretos comunes, aislamiento y
  retirada. No se hicieron llamadas de proveedor con credenciales reales ni
  revisión visual manual del escritorio.
- `python -m compileall` y `git diff --check` aprobados tras los cambios.

## Aún no implementado

- PostgreSQL no está disponible localmente; su integración y concurrencia se
  validan mediante GitHub Actions.
- Aún no están configurados el URL/clave de Supabase en un entorno de
  despliegue ni la contraseña de su PostgreSQL administrado. No se debe pegar
  la clave secreta en el chat; se configura únicamente como variable privada
  del backend.
- La API FastAPI no está desplegada. Supabase proporciona Auth y PostgreSQL,
  pero no aloja por sí solo este servidor Python; aún hace falta un servicio de
  hosting compatible para que los clientes externos usen la API comunitaria.
- Endurecimiento de producción de autenticación: límite de intentos y
  recuperación de cuenta. La verificación de correo se administra en Supabase.
- Detección de hardware en otros equipos y sistemas operativos; CUDA es
  opcional y no se presume.
- Verificación manual de la interfaz gráfica y publicación de instaladores
  Windows/Linux.
- Entrenamiento real con corpus autorizado, evaluación de calidad/seguridad y
  publicación del checkpoint. Los datasets congelados de conversaciones
  aportadas aún no alimentan el entrenamiento.
- Límites locales de CPU/RAM, controles de pausa/reanudación y sandbox
  reforzado para el nodo.
- Integración del libro mayor con trabajos verificados y autorización.
- Integración de modelos, chat o memoria.
- Rastreo web, RAG, uso de candidaturas en entrenamiento o motor de evolución.
- Interfaz web, instaladores de escritorio y operación de producción.
- TLS, rate limiting, CSRF si se añaden cookies, alertas y respuesta a
  incidentes antes de exposición pública.

Docker no está instalado o no está disponible en PATH en el entorno de
desarrollo actual; la verificación PostgreSQL se delega a GitHub Actions.

Las candidaturas se guardan en la base de datos configurada y esta versión no
cifra su contenido a nivel de campo. El filtro de secretos no es exhaustivo ni
es una detección de PII; no aportar contenido sensible. No hay un proceso que
entrene automáticamente. Las rutas administrativas de revisión y congelación
están protegidas por `AI_HOME_TRAINING_REVIEW_TOKEN`; el endpoint de revisión
expone contenido candidato a quien posea ese secreto. La API bloquea la
retirada de muestras ya congeladas y conserva snapshots aunque se borre la
cuenta de origen.

## Próximos criterios de salida

Antes de activar trabajos o créditos:

1. Probar migración inicial y consultas críticas contra PostgreSQL real.
2. Añadir rate limiting, política de sesión/credenciales y configuración TLS
   antes de cualquier exposición pública.
3. Implementar autorización en cada nuevo recurso de usuario.
4. Probar concurrencia, duplicación y recuperación del ledger.
5. Implementar política de tareas que impida distribuir datos privados.
6. Desarrollar cliente con permisos explícitos y límites locales.

Actualizar este archivo únicamente después de verificar los cambios descritos.

## Foundation Model — implementación inicial (2026-11-26)

Se añade un subsistema propio en `ai_home.foundation` con BPE desde cero, Transformer RoPE/GQA, entrenamiento, checkpoints e inferencia. El modelo byte-level y las funcionalidades anteriores se conservan. La integración de escritorio añade un proveedor independiente para checkpoints propios.

**Estado: parcial, mecánica neuronal verificada en CPU (2026-11-26).** La suite ejecutada obtiene 127 pruebas aprobadas y 5 omitidas (CUDA y PostgreSQL). El smoke entrena 27.808 parámetros desde cero durante ocho pasos, reanuda desde el paso cuatro, guarda/carga checkpoints y genera texto real todavía incoherente. Cambio L2 de embeddings tras reanudación: 0,02502243; pérdida de validación: 5,615235. PyTorch disponible: 2.15.0+cpu. La integración visual y distribución siguen sin demostrar; CUDA se verificó posteriormente en un entorno separado (ver CUDA_VALIDATION.md). Las restricciones de instalación descritas anteriormente ya no bloquean CPU; la publicación remota sigue pendiente.

Consultar [auditoría y matriz completa](FOUNDATION_AUDIT.md), [modelo](FOUNDATION_MODEL.md), [entrenamiento](TRAINING.md) y [pruebas](TESTING.md). Aprendizaje continuo, distribución, créditos de entrenamiento, RAG público, evolución y paneles completos siguen pendientes; no se reduce el objetivo ni se declaran demostrados.



## Integración HTTP Foundation — 2026-11-26

Implementada inferencia propia autenticada en el endpoint de chat, configuración de dispositivo/presupuesto/salida y exclusión mutua por proceso. Prueba real con entrenamiento y generación, rechazo de checkpoint corrupto/ausente y ausencia de fallback externo: 11 pruebas de chat aprobadas. Streaming HTTP, selección de versiones por usuario e interfaz de control completa siguen pendientes.

## Contribuciones hacia Foundation — 2026-11-26

Conectada exportación administrativa congelada a entrenamiento SFT local real. Verificados consentimiento, hash y revisión; el resultado es candidato de uso restringido, no publicación. Prueba extremo a extremo: crear contribuciones, aprobar, congelar, exportar, entrenar, guardar y recargar. Replay, fine-tuning incremental y promoción siguen pendientes.

## Aprendizaje incremental — 2026-11-26

Implementados candidatos desde checkpoints propios, BPE congelado, replay por lote, holdout fijo explícito, trazabilidad ancestral, bloqueo de contaminación exacta y propagación de restricciones. Experimento CPU real con informe en CONTINUAL_SMOKE_CPU.json. Promoción/rollback centralizado, replay restringido, capacidades lingüísticas y control general del olvido siguen pendientes.

## Evaluación y evolución experimental — 2026-11-26

Implementados probes de generación versionados y ciclo hipótesis/control/experimento/comparación. Ejecución real CPU: dos candidatos, diez probes cada uno, 0 aciertos y propuesta rechazada. No hay mejora de capacidades demostrada ni promoción automática. Ver EVOLUTION.md e informes JSON.

## Validación GPU completada — 2026-11-26

Entrenamiento, reanudación e inferencia propios verificados en RTX 4070 SUPER con FP32/FP16/BF16. Suite del entorno CUDA: 139 aprobadas, 4 omitidas (PostgreSQL). Checkpoint y métricas reales conservados. No se atribuye calidad conversacional. Comandos y límites: [CUDA_VALIDATION.md](CUDA_VALIDATION.md).

## Primer trabajo Foundation en nodos — 2026-11-26

Integrada tokenización BPE pública en scheduler/NodeAgent con capacidades declaradas, límites e integridad verificada. Dos procesos worker completaron dos shards por HTTP real en loopback. No equivale a entrenamiento neuronal distribuido; tampoco hay créditos ni verificación entre equipos todavía. Ver DISTRIBUTED_TRAINING.md.

## Entrenamiento neuronal con nodos — 2026-11-26

Integrado foundation.gradient.v1 con scheduler/NodeAgent y verificación por recomputación. Dos identidades de nodo aportan gradientes reales sobre shards públicos; agregación por tokens y AdamW producen checkpoints e inferencia. Demostración de dos rondas por HTTP loopback; solo un equipo y CPU. Créditos, multihost y escalabilidad de verificación pendientes. Ver DISTRIBUTED_TRAINING.md.

## Créditos Foundation — 2026-11-26

Reservas/liquidaciones/reembolsos integrados con el ledger en la misma transacción del trabajo. API de saldo y movimientos propios, cancelación, devolución por leases agotados y por eliminación de cuenta. Cuatro trabajos de gradientes reales pagaron 10 créditos a cada uno de dos usuarios en una base de prueba. Cobro de inferencia, interfaz y concurrencia PostgreSQL siguen pendientes. Ver CREDITS.md.


## 2026-11-26 — Account-scoped local memory

Desktop memories now use an explicit guest vault or a SHA-256 namespace derived
from the normalized server URL and authenticated account UUID. Login responses
include that stable UUID. Older servers lacking it are rejected by the desktop
login callback rather than falling back to a shared account vault. Existing
legacy notes remain untouched in `memory/`; no automatic ownership migration is
performed. New notes live in `memory-profiles/`.

Account transitions clear the conversation, draft, active memory references and
consent controls, and close private memory/contribution windows. Late chat and
training callbacks are discarded after conversation changes; background results
and errors are discarded after session-token changes. Clearing a conversation
does not cancel computation already running on the provider.

This is application-level separation, not filesystem encryption or protection
against another person/process using the same OS account. Guest notes belong to
the shared local guest profile. Memory context remains eligible for inclusion in
requests to the explicitly selected provider. Legacy migration, encrypted storage,
retrieval-role hardening and a live visual desktop review remain outstanding.

Validation: profile isolation including cross-profile link rejection, stable login
identity, stale answer/error suppression and private-window closure have automated
coverage. Full CUDA-enabled suite: 163 passed, 4 PostgreSQL tests skipped (no server).


### Public knowledge retrieval increment (2026-11-26)

Added a versioned reviewed-public TF-IDF index with provenance, bounded context
packing, and opt-in retrieval in authenticated Foundation chat. Retrieved text
and personal memories use user-level context, not system instructions. This is
retrieval, not weight training or a demonstrated improvement in model quality.
See [Knowledge retrieval](KNOWLEDGE.md) for commands,
permissions, tests and limitations. Network connectors and desktop references
remain outstanding.


### Desktop retrieval references (2026-11-26)

Connected the opt-in server retrieval request to the desktop chat. Source metadata
is validated and rendered in a read-only viewer separate from conversation text.
Account transitions reset retrieval selection and close private viewers. Training
submission of retrieval conversations is held pending a provenance-preserving
export/review workflow. Validation: 175 passed, 4 skipped; actual hidden Windows
Tk construction/viewer/clear/profile-switch check passed. See KNOWLEDGE.md.


### Separate knowledge-use permissions (2026-11-26)

Added a retrieval purpose to the shared dataset validator and loader. Public
retrieval-only documents can now be indexed without granting training permission.
The trainer and train/validation splitter still reject them. Explicit retrieval
denials override legacy training permissions; private reviewed conversation
exports remain excluded. Index update tests verify immutable earlier versions and
independent source metadata on the replacement version.

Validation: CUDA-enabled full suite 181 passed, 4 PostgreSQL tests skipped.


### Wikipedia structured connector (2026-11-26)

Preserved the existing importer outputs and added pending structured article
records. A review-template and approval CLI exports immutable versioned datasets
with separate training/retrieval permissions and review identity/hash. Approval
binds text and metadata; no automatic approval occurs. Verified a synthetic dump
through review and retrieval, plus rejection cases. Full suite: 189 passed,
4 PostgreSQL skipped; final connector-specific run: 9 passed. Documentation and
reproducible commands are in KNOWLEDGE.md. Large-dump review and real-source
license/privacy review remain operator work.


### Model lifecycle registry (2026-11-26)

Added local content-addressed candidate storage with real retained-checkpoint
evaluation, explicit approval gates, transactional activation history and rollback.
Foundation API serving can select the active approved registry version. Direct
checkpoint mode remains available and is not an approval claim. See
[Model registry](MODEL_REGISTRY.md) for commands, reproducibility evidence and limitations.


### Model approval regression gate (2026-11-26)

Subsequent registry approvals now compare compatible retained evaluations against
an approved baseline. The default is the active version (or most recent approval),
with explicit baseline selection available and recorded. Previously passing probe
regressions block approval even when total accuracy is unchanged. Category deltas
and gained/lost probe IDs are committed atomically with approval. Incompatible
suite/runtime settings fail closed. Existing approvals are preserved without
inventing historical evidence.

Validation: full CUDA-enabled suite 192 passed, 4 PostgreSQL tests skipped.


### Approved model API selection (2026-11-26)

Added authenticated approved-version catalog and optional per-request version
selection. Responses identify the model actually used; choosing a version does
not alter global activation. Candidate versions are excluded from the catalog and
cannot be selected. Registry resolution selects ID and checkpoint together.
Tests exercise two real checkpoints, explicit non-active selection, unchanged
activation, authenticated catalog and candidate rejection.

Validation: full suite 192 passed, 4 PostgreSQL skipped. After resolving schema namespace warnings, registry/chat subset: 13 passed without warnings.


### Desktop model version selection (2026-11-26)

Connected the approved server catalog to a read-only desktop selector. The first
registry response pins its actual UUID; subsequent requests retain that version
across server activation changes. Explicit response mismatches are rejected.
Conversation source metadata includes the version UUID, while chat/profile resets
clear the selection. Validation: 194 passed, 4 PostgreSQL skipped; real hidden
Windows Tk selector/reset checks passed. No global activation is performed by the
client and no visual layout certification is claimed.


### Resource-aware training configuration (2026-11-26)

Added explicit auto-fit CLI and reusable planner preserving effective batch via
microbatch/accumulation adjustment. Optional model shrinking requires its own flag
and is disallowed during resume. Plans are retained beside checkpoints. CPU and
CUDA training share the memory estimate. Real CUDA calibration identified and
corrected missing runtime overhead; repeated training and actual auto-fit CLI
execution succeeded. Evidence: RESOURCE_PLAN_CUDA.json; limitations in TRAINING.md.

Validation after calibration: 197 passed, 4 PostgreSQL tests skipped.


### Desktop credits view (2026-11-26)

Added read-only account balance/history view using existing authenticated ledger
endpoints. Shows available/reserved/earned values and signed postings, closes on
profile change, rejects malformed data, and guards late callbacks after window
closure. Verified actual funded ledger reads and separation from another account,
plus hidden real Windows Tk rendering with fixture values. Full suite: 199 passed,
4 skipped before the added ledger bridge test; final credit tests: 9 passed.


### Foundation streaming API (2026-11-26)

Added authenticated SSE generation with actual model-version/reference metadata,
text deltas and completion metrics. A bounded worker queue supports cancellation
and shares admission locking with ordinary inference. Private errors are redacted;
partial streams require a done event before being considered complete. Retrieval
context packing is shared between both routes. Validation: 202 passed, 4 skipped.
Desktop integration and real-network disconnect/latency tests remain pending;
see STREAMING.md for protocol details and cancellation limits.


### Real HTTP streaming cancellation (2026-11-26)

Added the cancellable client-side SSE consumer with version, ordering, completion
and response-size checks. Verified cancellation while waiting for headers and
rejection of incomplete streams. A real Uvicorn/HTTP loopback test trains a tiny
checkpoint, receives real generated text, cancels an in-flight stream, observes
server stop and succeeds on the next inference. The test controls pacing after
an actual generated chunk. Desktop widget wiring remains pending.

Validation: 210 passed, 4 PostgreSQL tests skipped; no warnings.


### Desktop progressive generation (2026-11-26)

Wired the cancellable SSE client to optional server-Foundation desktop mode.
A bounded preview is updated only through Tk's event queue; stop/clear/profile/
close cancel the stream. Only complete answers enter history or contribution data;
late fragments and completion-after-stop are discarded. Existing version pinning
and retrieval provenance remain intact. Validation: 212 passed, 4 skipped; hidden
real Windows Tk preview/stop-race/reset check passed with isolated UI fixtures.


### Bounded API generation controls (2026-11-26)

Exposed Foundation temperature, top-k, top-p, repetition penalty, seed and token
limit in both chat routes, returning effective settings with responses/metadata.
Strict validation rejects invalid values and requests above the server token cap.
External backends explicitly reject these Foundation-specific options. Actual
checkpoint tests compare greedy direct, complete-response and streaming outputs
with the same retrieved context. Desktop sampling widgets remain pending.

Validation: full CUDA-enabled suite 227 passed, 4 PostgreSQL tests skipped.


### Desktop generation settings (2026-11-26)

Added the optional Foundation generation dialog with six validated controls and
restoration of server defaults. Both response modes forward an immutable request
snapshot; profile changes reset settings. Local/external routing does not silently
ignore custom Foundation settings. Validation: 229 passed, 4 skipped; hidden real
Windows Tk apply/reset check passed. Visual layout inspection remains pending.


### Live desktop layout review and repair (2026-11-26)

Used computer-use screenshots of the actual Windows app to identify clipped chat
controls and inaccessible lower provider settings. Added scrolling to settings
pages, wrapped the memory explanation separately, made chat canvas sizing flexible,
and reserved status-bar space. Rechecked corrected chat and scrolled provider
controls visually; closed the isolated review instance. No account/credential
changes. Full suite: 229 passed, 4 skipped. Scope and remaining visual checks are
recorded in DESKTOP_VISUAL_REVIEW.md.


### Foundation checkpoint inspection interface (2026-11-26)

Added read-only desktop Model/Training views and CLI inspection of local Foundation
checkpoints. Counts derive from validated tensors; file/tokenizer hashes, actual
architecture and retained training metrics are exposed without tensor/optimizer
payloads. Inspection does not activate or approve models. Validation: 230 passed,
4 skipped; hidden real Windows Tk inspection of the CUDA experiment checkpoint
passed. Broader training/evaluation/evolution control panels remain incomplete.


### Desktop knowledge explorer (2026-11-26)

Added local reviewed-index inspection and query controls. The view reports verified
index version/counts and source license/permission metadata, distinguishing legacy
permission inheritance. Search displays actual fragments with provenance and never
runs generation or sends a query remotely. Profile/window lifecycle guards apply.
Validation: 231 passed, 4 skipped; hidden real Windows Tk index/search/profile-close
check passed using authored fixtures. Visual inspection of the new dialog remains.


### Dataset integrity at load time (2026-11-26)

The shared loader now verifies a declared dataset version against both the input
records and their canonical prepared representation. Supplied document hashes must
match normalized text, and an exported review hash must agree with every record.
Content, license or permission edits after export invalidate the retained version.
Reading is bounded while consuming the file, not only through a prior stat check.
Legacy unversioned datasets remain supported with normal provenance/permission
validation. Hashes establish consistency, not authenticated reviewer identity;
a party able to rewrite all hashes can still construct a different dataset.

Validation: 235 passed, 4 PostgreSQL skipped. Tests modify approved content,
licenses, permissions and review binding, and exercise the actual read budget.


### Controlled foundation training cancellation

Added cooperative cancellation at complete optimizer updates, resumable checkpoint
saving, CLI first-Ctrl+C handling and explicit completion/stop metadata in the
checkpoint inspector. CPU and CUDA tests verify resumed weights against an
uninterrupted run and cancellation before initialization creates no checkpoint.
Validation: 237 passed, 4 PostgreSQL tests skipped (server unavailable). The signal
handler was inspected; an interactive Windows Ctrl+C run has not been demonstrated.
See TRAINING.md for cancellation latency and second-interrupt limitations.


### Explicit vocabulary migration

Added an authorized-corpus CLI and Python migration path for append-only BPE
expansion. Preserves existing rows and other weights, initializes new rows from
a seed, retains privacy/holdout lineage, exclusively writes a separate candidate
and requires fresh optimizer training. Tests cover tied/untied models, exact
weight preservation, Unicode roundtrip, deterministic initialization, overwrite
and holdout rejection, resume rejection and real follow-up training.
Validation: full existing suite 237 passed, 4 PostgreSQL skips; after adding
migration cases the focused model suite passed all 12 tests. Initial new tests
used invalid warmup settings, corrected before the passing run. No quality
improvement or automatic promotion is claimed. See TOKENIZER.md for usage.


### Reproducible expanded-vocabulary training CLI

Added `--initial-checkpoint` and `--validation-dataset`; fresh optimizer training
is distinct from exact resume. Auto-fit uses the actual parent architecture and
cannot shrink it. Fixed the vocabulary CLI's dataset-loader tuple unpacking,
which Python API-only tests had not exercised. A real subprocess test now runs
vocabulary expansion followed by fresh CPU training with explicit holdout and
auto-fit, checks changed weights, preserved parent, matching architecture and
rejects conflicting resume/shrink options. Full suite: 240 passed, 4 PostgreSQL
tests skipped because the server is unavailable.


### Desktop evaluation/evolution report inspection

Added local read-only report viewing with pass totals or experiment decision,
full recorded metrics/configuration, bounded finite JSON loading and evaluation
summary consistency checks. Tests read the retained real CPU experiment reports
and reject inconsistent/oversized/invalid reports. A live Tk smoke loaded the
actual baseline report into the text widget; layout has not been visually
reviewed. Full suite: 245 passed, 4 PostgreSQL tests skipped. This provides report
inspection, not yet desktop evaluation/experiment execution controls.


### Evolution individual regression checks

Replaced aggregate-only acceptance checks with per-probe regression checks
against both control and baseline. Require matching execution protocols and
consistent individual/category evidence; reject non-finite control losses as
well as experimental losses. Reports retain the regressed probe IDs. Tests
include increasing aggregate accuracy that conceals a lost correct response,
protocol mismatches, inconsistent metrics and real two-candidate training.
Validation: 245 passed, 4 PostgreSQL skips. No acceptance or quality gain claimed.


### Desktop evaluation execution

Added checkpoint/suite/output selection, CPU/CUDA selection, generation budget,
background execution, per-probe progress and cancellation. Window/profile/app
closure requests cancellation; complete reports publish exclusively and partial
reports are discarded. Real training/evaluation tests verify progress, read-back,
output collision and pre/mid-run cancellation. A live Tk smoke invoked the actual
button, evaluated the existing CUDA-trained checkpoint on CPU and saved a real
10-probe report under artifacts/ui-review. Layout has not been visually reviewed.
Full suite: 246 passed, 4 PostgreSQL skips. No quality or promotion claim.


### Desktop real training and resume

Added local dataset/model/training configuration selection, CPU/CUDA device,
new run directory, resume checkpoint, actual step/loss/throughput and controlled
stop. Run directories retain settings and results. Training/evaluation workers
exclude each other in the desktop; normal close waits for the trainer to save.
A real Tk button smoke trained two CPU steps and verified changed embeddings
and a completed checkpoint under artifacts/ui-review. Functional cancellation
and resume tests compare all tensors against uninterrupted training. Full suite:
247 passed, 4 PostgreSQL skips. Dialog layout has not been visually reviewed.
Private SFT and fresh-parent training are still CLI-only; no automatic promotion.


### Desktop reviewed SFT and fresh-parent training

Extended the training form/job to distinguish public corpora and reviewed SFT
exports, random initialization/exact resume/fresh parent, and optional independent
validation. Existing authorization, consent, manifest, holdout and restricted
lineage checks remain enforced. The API-reviewed contribution integration test
now runs the desktop job backend for real SFT and a restricted child, and rejects
a causal configuration before creating a run directory. A live Tk smoke executed
fresh-parent training from the form and verified changed weights. Full suite:
247 passed, 4 PostgreSQL skips. Layout and private-SFT button interaction have
not been visually reviewed; no dataset or model has been published.


### Inference accounting foundation

Added migration 0007 and transactional reservation/settlement/refund primitives
for inference, owner-scoped idempotency, request/price binding and conditional
terminal transitions. No chat charge is enabled yet: API/stream lifecycle and
crash reconciliation integration remain outstanding. Tests cover insufficient
funds rollback, ownership, conflicts, repeated refunds/settlements, injected
posting failure rollback, two concurrent duplicate reservations and concurrent
settlement on file-backed SQLite. Migration upgrade/check/downgrade passed.
Full suite: 251 passed, 4 PostgreSQL skips. No real user funds were touched.


### Non-streaming inference credit integration

Connected optional flat Foundation completion pricing to authenticated API
requests, explicit per-request spending limits/UUIDs, transactional reservation,
success settlement and runtime-failure refund. Added price discovery, response
charge metadata and inference reservations in balance summaries. Duplicate keys
return 409 without repeat generation or debit. Paid streaming is explicitly
unavailable until its cancellation accounting is integrated. Default price is 0;
no deployment or real account was changed. Existing real-model API test now
verifies a genuine generated completion charged once, retry rejection and corrupt
checkpoint refund. Full suite: 251 passed, 4 PostgreSQL skips; final chat checks:
11 passed. Desktop consent, streaming and crash reconciliation remain outstanding.


### Streaming inference credit lifecycle

Connected streaming reservations to the acquired worker slot, terminal success
settlement and cancellation/failure refund. SSE metadata/final events disclose
reserved/charged credits, with explicit insufficient-funds and duplicate-key
errors. The real-model API test verifies successful streaming debit, duplicate
rejection and insufficient funds. The actual loopback Uvicorn/httpx test now
uses file-backed SQLite, disconnects after a real paid generated chunk, observes
the reserved balance and verifies full refund after worker shutdown. Full suite:
251 passed, 4 PostgreSQL skips. Crash reconciliation and desktop consent remain
outstanding; default pricing remains free.


### Desktop per-response credit authorization

Added server price discovery and an explicit per-response limit in the credits
window (zero by default, in-memory only, reset on account/profile transition).
Requests snapshot the limit and UUID for streaming or ordinary completion,
validate returned charge/identity and display consumed credits. No automatic
retry or total-session budget is implied. Client transport tests verify both
request forms and receipt rejection; privacy fixtures updated for the new state.
An initial source encoding issue and two outdated fixture failures were fixed.
Final full suite: 257 passed, 4 PostgreSQL skips. Credit dialog layout has not
been visually reviewed. Crash reconciliation remains outstanding.


### Expiring inference reservations and lazy crash recovery

Migration 0008 adds reservation deadlines (configurable 30–3600 seconds, default
600). Settlement conditionally rejects expired reservations. Balance reads and
new priced requests transactionally refund up to 100 expired reservations for
the authenticated owner; unexpired and other-owner reservations remain intact.
Recovery is lazy, not a global background service. Tests dispose/reopen database
connections to recover durable abandoned reservations, reject late settlement,
preserve live reservations, retain idempotency and verify API balance recovery.
The initial SQLite naive/aware ORM synchronization failure was corrected by
using database-only deadline predicates and refreshing the row. Final suite:
258 passed, 4 PostgreSQL skips; migration upgrade/check/downgrade passed.
No production database or real user balance was changed.


### Account deletion and pending inference credits

Account deletion now refunds reserved inference in the same local deletion
transaction while preserving settled charges and pseudonymous ledger evidence.
Paid reservation admission and deletion serialize through the user row, with a
fresh existence check after authentication. The deletion API test covers pending
and settled reservations, final balances, session revocation and rejection of a
late settlement attempt. Full suite: 258 passed, 4 PostgreSQL skips. Concurrent
account deletion/admission has not been exercised against PostgreSQL here.


### Actual PostgreSQL validation and concurrency correction

Downloaded publisher-checksummed portable PostgreSQL 17.12, initialized an
isolated localhost-only SCRAM cluster and ran the previously skipped integration
tests. Added real concurrent inference reservation/settlement/expiry coverage.
Found and fixed unreliable insert rowcount reporting on PostgreSQL: both same-key
callers could claim admission although the ledger debited only once. RETURNING
now determines admission, terminal transitions and account lock existence. Also
bound migration execution to the explicit test database URL. Final full suite:
263 passed, zero skipped. The test server was stopped afterward. Details and
reproduction are in POSTGRES_VALIDATION.md; no production database was touched.


### Cancellable evolution stages

Connected cancellation/progress through baseline probes, continual candidate
training, holdout comparison and candidate probes. Interrupted runs retain
checkpoints and an explicit phase record, omit the final decision and do not
start the remaining arm. The CLI handles first Ctrl+C cooperatively. Real-model
tests stop control training at step 1, reload its resumable checkpoint and
verify absence of comparison/final/experiment artifacts. Final suite: 258 passed,
5 PostgreSQL tests skipped because the portable server is stopped; prior full
PostgreSQL verification remains documented separately. Interactive CLI signal
entry has not been manually demonstrated.


### Desktop evolution execution

Added the experiment form and bounded file-entry job adapter, real stage/probe/
optimizer progress, controlled cancellation and app-close waiting. Local training,
evaluation and evolution workers exclude one another. The existing real two-arm
experiment test now exercises the desktop adapter. A live Tk button run trained
control and experiment (embedding deltas 0.1138595 and 0.0569225), then rejected
the proposal because experimental held-out loss did not match the control.
No promotion occurred. Evidence is under artifacts/ui-review/evolution-
dfa437b9c0a643f6897b01962b0f29f6. Layout has not been visually reviewed.
The first suite run overlapped the Tk run and hit transient host RAM guards in
three tests. Sequential rerun: 258 passed, 5 PostgreSQL skips (server stopped).
No memory guard was relaxed and no other application was stopped.


### Mixed CPU/CUDA distributed gradients

Added explicit CUDA worker execution with device-memory checks, preserving the
bounded public FP32 protocol and unchanged CPU verifier tolerances. Extended
NodeAgent and the two-worker HTTP demonstration with explicit device choices.
Actual mixed CPU/CUDA run completed two rounds, changed embeddings by 0.1838997,
reloaded a checkpoint and settled 10 synthetic credits per worker. Generated
text remains incoherent. CUDA tensor execution and CPU re-verification are tested.
Full suite: 259 passed, 5 PostgreSQL skips (server stopped). Evidence is retained
in DISTRIBUTED_MIXED_CUDA.json. No multi-host or multi-GPU performance claim;
resource-aware scheduler ranking remains incomplete.


### Live worker memory admission

NodeAgent now sends available RAM and optional CUDA free memory before claims.
Authenticated heartbeats validate snapshots and timestamp them on the server.
Scheduler admission considers installed/available RAM, stale snapshots and a
bounded-task CUDA memory floor. Legacy nodes retain prior behavior; workers
still enforce local memory guards and results remain independently verified.
Tests cover malformed CUDA reports, insufficient RAM/VRAM, stale resource
snapshots despite live claim heartbeats and successful refreshed admission.
Full suite before the added assertions: 259 passed, 5 PostgreSQL skips; final
node/distributed suites: 29 passed. Performance/network-aware ranking remains
outstanding; these are advisory snapshots, not reserved or attested resources.


### Renewable node job leases

Added authenticated lease renewal with an atomic owner/token/status/expiry
predicate. Workers renew on a bounded interval while a handler runs, stop/join
the helper on completion or exception, and discard results after renewal failure.
Tests verify token/owner rejection, expiry fencing, no attempt inflation and
worker success/failure behavior with an actual renewal thread. Full suite:
262 passed, 5 PostgreSQL skips (portable server stopped). Renewal alone grants
no credit and does not freshen resource snapshots. Handler/kernel interruption
and multi-host long-duration validation remain outside this completed increment.


### Evaluation checkpoint identity binding

Fixed a report provenance race: evaluation previously hashed the path after
generation, which could identify different bytes than those loaded. Safe loading
now hashes one open handle before/after deserialization and exposes the verified
source identity on the runtime model. Evaluation reports use that identity.
Tests modify a loaded path between real probes and modify file bytes during
loading, verifying correct attribution and rejection respectively. Full suite:
264 passed, 5 PostgreSQL skips (server stopped). This adds no signature or
publisher-authentication claim.

### Own-model feature extraction

Added final hidden-state access without changing logits, plus bounded windowed
mean embeddings from an own-model checkpoint, with source/checkpoint/tokenizer
identity. CPU/CUDA tests compare against explicit full-token state pooling and
check deterministic output, input budgets, and unchanged logits. Full CUDA-env
suite: 266 passed, 5 skipped (optional PostgreSQL server stopped). TF-IDF remains
the retrieval backend; neural indexing and retrieval quality are still pending.

### Optional checkpoint-bound neural retrieval

Connected own-model embeddings to immutable public knowledge indexes, CLI build /
search and generation-context retrieval. Stored vector identity must match the
loaded checkpoint and tokenizer; normalized shape/count/budget validation applies.
Existing TF-IDF remains the default. CPU and CUDA tests cover exact-text retrieval,
provenance, checkpoint mismatch, malformed vectors and RAG context refusal. Full
suite: 266 passed, 5 PostgreSQL-dependent skips; final added context assertion:
2 embedding tests passed on CPU/CUDA. No semantic retrieval improvement is claimed.

### Retrieval comparison with labeled paraphrases

Added a bounded retrieval evaluation CLI with per-query document relevance,
recall@k and reciprocal rank, plus eight original fictional ES/EN documents and
separate paraphrase queries (retrieval permission only). Actual CPU comparison
with the existing trained Nano checkpoint: both TF-IDF and own neural retrieval
hit 5/8 at k=3; MRR 0.5000 versus 0.5417. This small diagnostic does not demonstrate
a general semantic gain. Reports and reproduction are in RETRIEVAL_COMPARISON.json
and RETRIEVAL_EVALUATION.md. Full suite: 267 passed, 5 PostgreSQL skips. The final
fixture wrapper change passed its focused test and both documented CLI paths.

### Desktop access to neural knowledge indexes

Added checkpoint/device controls to the local knowledge explorer and optional
checkpoint-backed inspection API. Full suite: 267 passed, 5 PostgreSQL skips;
additional CPU/CUDA inspection assertions: 2 passed. A real Tk widget smoke used
the existing neural index/checkpoint, entered a query, invoked the search button,
verified returned neural identity/provenance, and checked controls re-enabled.
That smoke used a synchronous callback harness; it does not prove background
thread behavior or visual layout quality. No semantic-quality claim is added.

### Knowledge creation/update flow in the desktop

Added creation of a new immutable knowledge index from authorized JSON, optional
own-model embeddings and CPU/CUDA selection, with cancellation and background
completion/error handling. Publication now exposes only flushed complete bytes,
uses exclusive hard-link creation and preserves existing versions. Permission and
privacy checks precede embedding work. Tests cover cancelled jobs, publication
failure cleanup, immutable destinations and cancellation after real CPU/CUDA
embedding computation. Full suite: 268 passed, 5 PostgreSQL skips; final added
embedding cancellation assertions: 2 passed. A real Tk smoke using actual worker
threads/event callbacks built the eight-document neural artifact at
`artifacts/ui-review/knowledge-acb6baf2e5774340ae1e4668d395d295.json`.
This verifies the functional flow, not visual layout quality or retrieval gains.

### Local model registry desktop control

Added existing-registry listing, candidate evaluation inspection, fixed-default
approval, approved-version activation and rollback, with operator/reason auditing.
All actions reuse the registry gates; no threshold override is exposed. Full suite:
268 passed, 5 PostgreSQL skips. A real Tk asynchronous smoke exercised rejection
of an insufficient candidate, two activations and rollback to the first candidate,
and verified the persisted audit sequence/operator in an isolated registry:
`artifacts/ui-review/registry-eed024a49ac24a6c9557693172bdc019/registry`.
The fixture used a newly trained one-step checkpoint and explicit zero-threshold
CLI approvals for two lifecycle-only test versions. The desktop itself retained
100% approval and rejected the third candidate. An older smoke checkpoint was
first rejected for missing required training lineage; that gate was not weakened.
No production registry was touched. Functional UI testing is not a visual review.

### Measured local model sizing

Added a bounded synthetic-token compute benchmark with actual AdamW updates,
synchronized CUDA timing, working-set preflight, device identity, sampled RSS,
peak CUDA allocation/reservation and nonzero parameter-change evidence. Ran Nano
(803,968 parameters) on CPU/CUDA and Micro (4,361,472) / Small (22,815,232) on CUDA
in separate processes. Raw measurements and configurations are committed; see
COMPUTE_BENCHMARKS.md for throughput, memory and scope limitations. No language
capability, optimal sizing, sustained performance or multi-GPU support is inferred.
Medium/Large sizing remains unresolved. Full suite: 270 passed, 5 PostgreSQL skips.

### Estado consolidado y entrega remota

El inventario vigente está en FOUNDATION_STATUS.md: conserva todas las áreas del
objetivo, diferencia evidencia de limitaciones y enumera las próximas líneas de
implementación. README, ROADMAP, TESTING y ARCHITECTURE enlazan ese inventario para
no presentar notas históricas como estado actual. El objetivo no está completado.
La rama se publicó en la PR borrador #1; main permanece intacta. CI de código
2e3d4c4 verificada: Python 3.11/3.12 con PostgreSQL, 267 correctas y 8 omisiones por
CUDA por versión; Foundation CPU, 262 correctas y 13 omisiones opcionales, más
smoke real y artefacto. Se corrigieron dependencia torch ausente en la matriz y
una carrera del test de cancelación, sin relajar gates ni omitir esos tests.

### Primer conector público de conocimiento por red

Implementado github-text-v1 con commit inmutable, host fijo, descarga acotada,
sin credenciales/redirecciones y registro extensible de conectores instalados.
Produce candidatos pendientes con hashes de bytes/texto, licencia declarada no
verificada y permisos de entrenamiento/recuperación desactivados. Integrado con
la revisión existente; las pruebas verifican permisos separados y rechazo de
fuentes mutables, rutas inválidas, redirecciones, HTTP fallido y exceso de bytes.
Descarga real de dos documentos AI-Home: 24.891 bytes JSONL; evidencia en
GITHUB_CONNECTOR_SMOKE.json, sin aprobación automática. Suite Windows/CUDA:
280 correctas, 5 PostgreSQL omitidas. Documentación en KNOWLEDGE_CONNECTORS.md.

### Publicación completa de artefactos de conocimiento

Unificados candidatos descargados, plantillas de revisión, datasets aprobados e
índices en una primitiva de escritura/flush y publicación exclusiva. Se evita
exponer destinos parciales si falla la escritura; dos publicadores concurrentes
conservan exactamente un archivo completo. Las pruebas inyectan fallo de flush,
cancelación previa y carrera de publicación. Suite local: 282 correctas y 5
PostgreSQL omitidas. La matriz PostgreSQL de la revisión anterior 7ee0a90 terminó
correctamente en Actions (run 45813703815); esta modificación tiene validación
local y todavía no se atribuye a aquella ejecución remota.

### Recuperación del punto de agregación distribuida

Añadida operación/CLI que reconstruye una ronda desde checkpoint público y sobres
payload/resultado persistidos, recomputa gradientes, restaura AdamW y publica una
nueva versión de forma exclusiva. Los reintentos idénticos conservan el resultado;
solicitudes distintas no pueden ocupar la misma ruta. No hereda validaciones del
padre como si evaluaran al hijo. Prueba de interrupción antes de publicar y smoke
HTTP con workers CPU/CUDA: pesos recuperados exactamente iguales a los originales,
retry sin doble actualización. Evidencia: DISTRIBUTED_RECOVERY.json. Suite completa:
283 correctas, 5 PostgreSQL omitidas; ajuste final de metadatos de dataset probado
con test enfocado y reconstrucción real. Reinicio automático del coordinador y
recuperación de descubrimiento/envío de jobs siguen pendientes.

### Coordinador reanudable con jobs y créditos idempotentes

Añadidos plan inmutable y CLI advance para reencontrar jobs tras reinicio con claves
deterministas vinculadas a servidor, cuenta, padre, shards y recompensa. Rechaza
cambios de identidad y conserva el opt-in explícito. Integra commit_round al acabar
los jobs; una pasada devuelve waiting o committed, sin daemon/polling ilimitado.
Test con FastAPI/DB/ledger real pierde la respuesta tras crear el segundo job,
reenvía y recupera ambos IDs, comprueba reserva única, procesa gradientes reales y
verifica reintentos sin doble actualización ni nuevos jobs. Suite: 284 correctas,
5 PostgreSQL omitidas; comprobación final del gate de padre: test enfocado correcto.
Reinicio automático de servicios, failover y prueba multihost siguen pendientes.

### Reanudación real del coordinador entre procesos

El smoke HTTP admite --restart-coordinator: cuatro procesos CLI independientes
recuperan/consultan/guardan/reconsultan la segunda ronda. Ensayo real CPU/CUDA con
recompensa de tres créditos por job: mismos IDs y reservas al reiniciar, pesos
idénticos a referencia, bytes inmutables al reintentar, cuatro jobs totales y seis
créditos ganados por cada worker. Evidencia: COORDINATOR_RESTART_SMOKE.json.
Suite local posterior: 284 correctas, 5 PostgreSQL omitidas. CI del código anterior
0029fc3 pasó en runs 45815026730 (matriz) y 45815026912 (Foundation). El ensayo nuevo
no demuestra reinicio del servidor, supervisor automático ni failover multihost.

### Supervisión acotada de rondas

El CLI advance admite --wait-seconds / --poll-seconds y la API supervise admite
cancelación cooperativa. Reintenta únicamente errores de transporte con el mismo
plan, detiene ante errores HTTP/política y devuelve observación/plazo/intentos sin
cancelar implícitamente jobs financiados. Deadline entre pasadas, no interrupción
forzosa de llamadas o kernels. Pruebas con reloj virtual de control y reconciliación
real API/ledger/checkpoint; suite completa: 285 correctas, 5 PostgreSQL omitidas.
No se presenta como servicio del sistema ni supervisor multihost.

### Correcciones voluntarias con procedencia y SFT

Nueva API owner-only de corrección: consentimiento renovado, revisión de derechos
/privacidad, candidato independiente pendiente y prefijo hasta la respuesta
corregida. Preserva original, vincula ID/hash de respuesta reemplazada, no incorpora
turnos derivados posteriores ni evita la revisión. Migración 0009 añade metadatos
nullable; congelación/exportación los vincula al manifest y SFT conserva ámbito
restringido. Test extremo a extremo entrena pesos reales; incluye acceso cruzado,
secretos, permisos, duplicados, revisión y compatibilidad de exportaciones previas.
Suite: 286 correctas, 5 PostgreSQL omitidas; comprobación final correcciones/revisión/
migración: 10 correctas. No se migró producción. Ver TRAINING_CORRECTIONS.md.
Interfaz de edición, contribuciones del propio proveedor y preferencias pendientes.

### Procedencia Foundation en contribuciones consentidas

El contrato de contribución admite ai-home-foundation con digest SHA-256 exacto
como modelo y URI Foundation coincidente. No acepta nombres mutables ni URLs de
proveedores para esa procedencia. Mantiene consentimiento, derechos y revisión.
Test con checkpoint realmente entrenado y respuesta generada enlaza contribución
y corrección; el servidor no confunde una declaración del cliente con atestación
remota. Suite: 287 correctas, 5 PostgreSQL omitidas. CI PostgreSQL de la revisión
anterior dfb7c70 (incluye migración 0009) pasó en run 45816699410. Falta transportar
la identidad cargada en el escritorio antes de habilitar su envío Foundation.

### Identidad cargada en contribuciones Foundation de escritorio

El adaptador devuelve respuesta real y digest ligado a los bytes cargados. El
escritorio fija esa identidad durante la conversación, rechaza sustitución del
checkpoint en el siguiente turno y utiliza digest/URI Foundation al aportar con
consentimiento. Limpiar chat/perfil borra el pin. Los chats proxy siguen excluidos.
Prueba real sustituye el archivo durante generación y verifica identidad original
+rechazo del siguiente turno. Smoke Tk real verifica generación, pin, contrato de
envío y reset; captura/valida el envío sin red, no se presenta como prueba HTTP.
Suite: 288 correctas, 5 PostgreSQL omitidas. No cambia permisos ni gates de revisión.

### Editor de correcciones en escritorio

Mis aportaciones permite cargar una contribución propia, elegir una respuesta,
revisar el prefijo y enviar un reemplazo con consentimiento/revisión renovados.
Nuevo UUID estable por editor, candidato pendiente, original intacto y callbacks
inválidos descartados al cerrar/cambiar perfil. Smoke con Tk real y API/DB en
proceso verifica ausencia de envío sin confirmaciones y nueva corrección con
procedencia. El test de correcciones/SFT atraviesa ahora el helper HTTP de escritorio.
Suite previa al último ajuste del test: 288 correctas, 5 PostgreSQL omitidas;
tests finales de correcciones: 2 correctos. Revisión visual pendiente. CI del código
anterior 27d005d pasó (Foundation y matriz PostgreSQL 3.11/3.12).

### Revisión visual del editor de correcciones

Revisión Windows con Tk real y cuatro mensajes de desarrollo: controles legibles
en el tamaño inicial, cambio de respuesta/prefijo correcto y error visible al
enviar sin confirmaciones. Sin envío de contribuciones en esta revisión. Ventana
cerrada y ausencia comprobada. Redimensionado/DPI y textos largos sin verificar;
detalle en DESKTOP_VISUAL_REVIEW.md. Suite completa actual: 288 correctas,
5 PostgreSQL omitidas, 39,48 segundos. No demuestra calidad del modelo.

### Paquete redistribuible Nano smoke

Añadido empaquetador/verificador acotado para el experimento propio: pesos con
estado de reanudación, BPE, configuraciones, corpus sintético público exacto,
reporte, licencia y manifiesto SHA256. Publicación exclusiva, sin extracción de
rutas ZIP; comprueba identidad del corpus/división, checkpoint y metadatos.
No acepta otros corpus ni candidatos heredados/restringidos y no equivale a firma
o aprobación de calidad. Actions construye, verifica y sube el paquete por separado.
Suite local: 291 correctas, 5 PostgreSQL omitidas, 46,50 segundos. Incluye pruebas
con entrenamiento real y rechazos por corrupción, metadatos, rutas y privacidad.
CI previa (9930d35): Foundation y PostgreSQL Python 3.11/3.12 correctos.

### Reproducción desde instalación aislada

Checkout fijado a 1f12fd8, venv Windows nuevo, PyTorch 2.15.0+cpu y wheel del proyecto
sin modo editable. Ejecución fuera del checkout con Python -I: entrenamiento,
reanudación, carga, generación, empaquetado y verificación correctos; pip check
sin dependencias rotas. También descargado y verificado en ese entorno el paquete
real producido por Linux CI. Documentados hashes, reporte, comandos y límites en
CLEAN_REPRODUCTION.md. No es una instalación integral de API/GUI/distribuido ni
una máquina nueva. CI de 1f12fd8: Foundation 283/13 omitidas; PostgreSQL Python
3.11 y 3.12, 288/8 omitidas cada uno. Inventario y encabezado TESTING actualizados.

### API instalada y migraciones incluidas en wheel

Detectado y corregido empaquetado: historial Alembic trasladado sin modificar
revisiones/operaciones a ai_home.migrations. Configuración histórica compatible y
nuevo python -m ai_home.migrate upgrade/current/check, sin depender del checkout.
Wheel instalado en venv aislado: migraciones reales, registro/login, 401 sin sesión,
inferencia y SSE por Uvicorn/HTTP loopback correctos; cierre del servidor verificado.
Script reproducible scripts/verify_installed_api.py ejecutado dos veces con DB
nuevas, informe sin credenciales en INSTALLED_API_SMOKE.json. Suite completa:
292 correctas, 5 PostgreSQL omitidas. Añadida verificación wheel/HTTP a CI; pendiente
resultado remoto. Limpieza de build/ rechazada por política automática; directorio
local generado excluido del commit, sin afectar código ni prueba.

### Distribuido desde wheel con aislamiento

Smoke distribuido usa migraciones empaquetadas y Settings explícitos; workers y
coordinador heredan -I. Prueba adicional garantiza que las migraciones explícitas
no afectan a la URL de otra DB del entorno. Wheel reinstalado en venv CPU nuevo:
4 procesos worker / 2 rondas reales HTTP, cambio L2 0,1838996708, dos cuentas con
6 créditos ganados cada una, requester sin reserva pendiente. Reinicios conservan
trabajos/checkpoint: waiting/waiting/committed/committed. 26,57 segundos; un solo
host, sin afirmar red multihost ni calidad. Reporte INSTALLED_DISTRIBUTED_SMOKE.json.
Suite: 293 correctas, 5 PostgreSQL omitidas (38,50 s). Nueva etapa CI para este
smoke y reportes; pendiente resultado. CI anterior 9b126d9 pasó, incluida API
instalada (run 45820882944), y PostgreSQL Python 3.11/3.12.

### Mediciones reales previas a selección por rendimiento/red

NodeAgent registra una muestra solo tras resultado aceptado: duración monotónica
de heartbeat/claim/ejecución/submit, bytes de cuerpos HTTP y tasas de servicio;
gradientes incluyen tokens/s. Limpia muestras al iniciar cada intento, también
si queda ocioso/falla. Sin contenidos ni credenciales. Node smoke/distribuido
persisten muestras en reportes. No altera todavía FIFO, créditos ni validación.
Experimento CPU/CUDA real de dos rondas y reinicio: 14,53–14,92 tokens/s CPU y
27,23–27,52 CUDA, incluyendo preparación de tareas diminutas. Reporte y límites
en WORKER_MEASUREMENTS.md/WORKER_MEASUREMENTS_SMOKE.json; no se presenta como
ancho de banda físico ni benchmark general. Suite: 293 correctas, 5 omitidas,
41,49 s. CI anterior f509371 pasó, incluido distribuido desde wheel (45821316226).
Falta transportar/validar muestras frescas y aplicar restricciones explícitas
antes de afirmar selección por rendimiento/ancho de banda/latencia.
