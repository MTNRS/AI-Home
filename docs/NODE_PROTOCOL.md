# Protocolo inicial de nodos voluntarios

Este protocolo todavía es experimental. En esta versión solo está habilitado
el trabajo tipado `math.sum.v1`; no hay ejecución de código arbitrario ni
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

