# Protocolo inicial de nodos voluntarios

Este protocolo todavía es experimental. Están habilitados
`math.sum.v1` y `foundation.tokenize.v1`; no hay ejecución de código arbitrario ni
integración de créditos. Los trabajos compartidos con nodos de terceros pueden
revelar sus entradas: el creador debe otorgar consentimiento explícito y no
enviar datos personales, secretos ni valores confidenciales.

## Registro

1. Una persona autenticada registra el nombre del nodo y el informe de
   hardware del dispositivo.
2. El servidor devuelve un token de nodo aleatorio una sola vez. El cliente
   debe guardarlo en el almacén seguro del sistema operativo; no se guarda el
   token en claro en la base de datos.
3. La persona puede consultar o revocar sus propios nodos. La revocación
   invalida el token y devuelve los trabajos asignados a la cola.
4. No se combinan la memoria de varias GPU: cada dispositivo tiene su propio
   índice, nombre y VRAM reportada.

## Trabajo permitido

- `math.sum.v1` acepta una lista acotada de enteros estrictos y devuelve su
  suma.
- El cliente ejecuta un handler escrito y versionado en el paquete. No recibe
  módulos, scripts, comandos, URLs arbitrarias ni una ruta para cargar código
  generado por usuarios.
- Crear el trabajo exige `allow_public_node_processing: true`. Se guarda la
  versión y la hora del consentimiento para auditoría.
- El servidor vuelve a calcular y valida el resultado antes de marcarlo como
  completado.

## Ciclo y endpoints

- `POST /api/v1/nodes`: registrar con una sesión de usuario.
- `GET /api/v1/nodes`: listar solo nodos propios y no revocados.
- `DELETE /api/v1/nodes/{node_id}`: revocar un nodo propio.
- `POST /api/v1/node/heartbeat`: el cliente informa que sigue activo.
- `POST /api/v1/jobs`: crear un trabajo con consentimiento e idempotency key.
- `GET /api/v1/jobs`: consultar solo los trabajos propios.
- `POST /api/v1/node/jobs/claim`: obtener un trabajo mediante un lease
  temporal. Un `204` significa que no hay trabajo disponible.
- `POST /api/v1/node/jobs/{job_id}/result`: enviar el resultado con el token
  del lease. Un lease expirado no se acepta.

Los tokens bearer solo deben usarse sobre HTTPS en redes no confiables. El
servidor aún no está preparado para exponerse a Internet y no incluye
limitación de intentos ni infraestructura TLS.

## Scheduler

Los leases vencidos se reencolan; tras tres intentos se marca el trabajo como
fallido. PostgreSQL es necesario para verificar los bloqueos y reclamaciones
concurrentes (`FOR UPDATE SKIP LOCKED`). La suite SQLite valida la transición
secuencial, pero no prueba seguridad de concurrencia multiworker.

## Próximos ejecutores

Embeddings e inferencia local se incorporarán como handlers versionados
independientes, después de seleccionar dependencias/modelos y revisar sus
licencias. No se consideran disponibles mientras no existan y pasen pruebas
reales.


## Extensión Foundation — 2026-11-26

RegisterNodeRequest acepta supported_job_types (allowlist). Si se omite, conserva
math.sum.v1. El worker actual anuncia también foundation.tokenize.v1. El tipo
real se incluye al asignar el lease; no se envían trabajos nuevos a clientes
antiguos. Los resultados permiten enteros o cadenas acotadas; sum sigue
exigiendo entero estricto. El servidor valida exactamente según el tipo.

Los trabajos de tokenización se crean en /api/v1/jobs/foundation/tokenize,
con consentimiento explícito, corpus público revisado y BPE acotado.
Idempotencia por propietario/clave conserva el trabajo o rechaza payloads
contradictorios. La finalización bloquea la fila en PostgreSQL y verifica lease,
nodo y artefacto antes de marcar completado. Los resultados solo aparecen en
la lista del propietario. No hay recompensas todavía. Detalles y demo real en
DISTRIBUTED_TRAINING.md.

## Trabajo neuronal opcional

foundation.gradient.v1 exige activación del operador, opt-in del worker, runtime PyTorch declarado compatible y presupuesto de RAM. Se conserva el protocolo de leases/idempotencia/propiedad. El servidor verifica gradientes por recomputación antes de aceptar. Los clientes antiguos continúan recibiendo solo sus tipos anunciados. Detalles, límites y evidencia en DISTRIBUTED_TRAINING.md.


## Live resource heartbeat

POST `/api/v1/node/heartbeat` accepts an optional JSON body with strict integer
`ram_available_bytes`, `foundation_device` (`cpu` or `cuda`) and, for CUDA only,
`cuda_free_bytes`. NodeAgent sends this snapshot immediately before every claim,
using psutil and CUDA memory reporting. The server timestamps it; claim liveness
updates do not freshen an old resource snapshot. Registration cannot set a trusted
resource timestamp. Owner node listings expose the last snapshot and timestamp.

When a snapshot exists, admission uses the smaller of installed and available
RAM. Stale snapshots block tokenization/gradient work until refreshed. Gradient
work requires at least 1 GiB RAM and CUDA-mode workers must report at least
256 MiB free VRAM (a conservative threshold for the bounded reference task).
Legacy nodes without snapshots retain the prior installed-RAM admission checks;
new workers still check actual memory locally before executing. This is advisory
self-reported eligibility, not hardware attestation, reservation of memory or
performance ranking. Independent artifact verification remains mandatory.


## Lease renewal during computation

POST `/api/v1/node/jobs/{job_id}/renew` requires the node credential and a
`lease_token` body field. A conditional database update extends only an active,
unexpired lease assigned to that node with that exact token. Expired, cancelled,
reassigned and wrong-owner/token requests return 404; renewal cannot resurrect
a lease. It also refreshes node liveness, but not its memory snapshot.

NodeAgent renews every 30 seconds during computation (configurable 1–60 seconds
for tests/deployments, below the 120-second lease window). A failed renewal stops
further renewals and prevents result submission. It cannot interrupt an active
CUDA kernel or handler; local compute may finish and be discarded. The renewal
thread is joined before submission/return, including handler exceptions.
Network calls remain bounded by the node client's configured timeout. Existing
result validation and reward settlement are unchanged. No credits accrue for
renewal alone; the owner can still cancel the job.
