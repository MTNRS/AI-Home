# Sonda de transferencia HTTP del nodo

`network_probe.measure_transfer(client, size=65536)` utiliza un cliente HTTP ya
autenticado como nodo. Ejecuta tres GET vacíos y una descarga/subida de contenido
público fijo, verificando tamaño y SHA256. El endpoint GET/POST es
`/api/v1/node/transfer-probe`; límite 1 MiB por transferencia, sin compresión de
respuesta, sin ejecutar código, persistir el cuerpo ni conceder créditos.

La función y NodeAgent.measure_transfer son explícitas; todavía no se ejecuta automáticamente desde el nodo ni
se utiliza como criterio del scheduler. El propietario del cliente configura su
timeout y transporte. No hace descubrimiento de hosts ni envía datos del equipo.
La respuesta de descarga se consume por bloques acotados y rechaza exceso.

Las métricas son tiempos y tasas del servicio HTTP de transferencia, separados
de ejecutar/verificar trabajos neuronales. Incluyen autenticación, servidor,
HTTP y transporte: **no son ancho de banda físico ni RTT exclusivamente de red**.
No se restan tiempos para inventar una medida de enlace. El payload repetitivo es
conocido y esta sonda no acredita trabajo útil ni protege contra nodos que mientan.

Validación inicial: `test_network_probe.py`, 2 passed in 2.64s. TestClient verifica
flujo real del endpoint, credenciales, integridad, tamaños y rechazo de codificación.
Estas pruebas en proceso no son un benchmark de red. Pendiente: ejecución HTTP
real, integración explícita de nodo, limitación de frecuencia/persistencia y uso
del scheduler con límites y caducidad; medición entre equipos físicos pendiente.

## Ejecución HTTP real

```powershell
python -m ai_home.network_probe_smoke --output-dir artifacts/probe-new
```

Crea una base SQLite migrada, servidor Uvicorn en puerto loopback efímero, cuenta
sintética y nodo con credencial. NodeAgent mide 64 KiB y 1 MiB; verifica hashes y
comprueba que no cambian los recuentos de trabajos/transacciones. Cierra cliente,
servidor, socket y motor; falla si el servidor no termina. Nunca imprime credenciales.
Exige directorio nuevo y conserva DB de prueba e informe sin tokens en report.json.

Ejecución Windows real `artifacts/network-probe-http-v1` pasó: trabajos 0→0,
transacciones 0→0. Informe HTTP_TRANSFER_SMOKE.json. Tiempos HTTP vacíos de esta
corrida aproximadamente 1.49–3.85 ms; no es RTT puro. CI añade el ensayo del paquete
instalado con Python -I y publica informe; resultado remoto de esta adición pendiente.
La evidencia es exclusivamente un equipo y loopback; no prueba red multihost.

## Presupuesto de frecuencia por proceso

El endpoint admite hasta 10 peticiones por nodo en ventana móvil de 60 segundos,
1000 peticiones totales y 64 MiB reservados por proceso. El upload reserva 1 MiB
antes de leer; fallos consumen su reserva. La respuesta de exceso es 429 con
Retry-After: 60. Un lock protege la admisión concurrente y el registro en memoria
queda acotado a 1000 eventos. La sonda de dos tamaños usa exactamente diez peticiones.

No es una cuota distribuida: reiniciar el proceso borra la ventana y varios
workers tienen presupuestos independientes. Para despliegue multihost sigue
pendiente un límite compartido; no sustituye la protección del proxy/servidor.

Pruebas: 4 passed in 1.13s, límite HTTP, admisión concurrente, presupuesto de bytes
y expiración. El primer intento detectó uso incorrecto de id en vez de node_id;
corregido antes de repetir la suite. Smoke HTTP repetido con éxito en
artifacts/network-probe-http-v2, sin cambios en trabajos/créditos.

## Publicación explícita de la muestra

`NodeAgent.measure_transfer(size, publish=True)` envía solo tamaño y tiempos crudos
a POST `/api/v1/node/transfer-measurement`. El esquema rechaza campos extra,
valores no finitos, tiempos fuera de (0,60] segundos, tamaño fuera de 1–1 MiB y
listas de RTT distintas de tres elementos. El servidor deriva tasas y mínimo HTTP,
añade received_epoch/expires_epoch (120 s) y conserva únicamente la última muestra
en hardware.transfer_measurement del nodo autenticado. Consume cuota de sonda.
No permite elegir otro nodo ni fecha, y no altera créditos/trabajos.

Los tiempos siguen siendo declarados por el cliente; no están vinculados a un
recibo verificable de transferencia. Se pueden falsear. La caducidad es metadata
para futuros consumidores; todavía no afecta al scheduler. La telemetría no es
prueba de ejecución ni autoridad de recompensas.

5 pruebas pasaron en 1.23s: validación, aislamiento de nodos y tiempos del servidor.
Smoke HTTP real `--publish` pasó con un payload 1 MiB (seis peticiones) en
artifacts/network-probe-published-v1; reporte HTTP_TRANSFER_PUBLISHED_SMOKE.json.
No hubo creación de trabajos/transacciones. La modalidad sin --publish conserva
las dos transferencias originales y no envía telemetría.

## Restricciones opcionales del scheduler

GradientSchedulingRequirements admite `min_transfer_download_bytes_per_second`,
`min_transfer_upload_bytes_per_second` y `max_transfer_http_round_trip_seconds`.
El último compara el mínimo de las tres observaciones HTTP, no un percentil ni el
peor caso. La elegibilidad exige muestra válida recibida hace como máximo 120 s y
recalcula tasas desde los tiempos crudos. Ausente/caducada/inválida solo bloquea
trabajos que pidan estos campos; los demás conservan las restricciones existentes.

Se mantienen los requisitos de medida de gradiente comparable, recursos recientes
y perfil del modelo, incluso si solo se especifica una restricción de transferencia.
Por tanto se necesita primero un trabajo de cómputo sin restricciones y una sonda
publicada explícitamente. No se ha activado sondeo automático. Los criterios se
incluyen en la idempotencia/plan existentes y aparecen con nombres en el visor.
Estas observaciones no atestadas pueden falsearse; no se usan para pagar créditos.

14 pruebas distribuidas/coordinador/sonda pasaron en 5.76s: umbrales de ambas tasas
y tiempo, muestra ausente/caducada y aceptación. PostgreSQL real local: siete
pruebas pasaron en 4.77s, incluyendo predicado JSON de las nuevas restricciones.
Pendiente demostración completa con sonda HTTP real seguida de asignación y
entrenamiento; estas pruebas usan tiempos controlados para decidir los umbrales.

## Flujo completo con gradientes reales

```powershell
python -m ai_home.foundation.distributed_smoke --output-dir artifacts/transfer-round-new --rounds 2 --reward-credits 3 --worker-devices cpu cuda --restart-coordinator --transfer-scheduling
```

Primera ronda sin restricciones obtiene medidas de cómputo. Antes de reclamar la
segunda, cada worker realiza/publica una sonda HTTP real. La segunda ronda exige
mínimos de transferencia de 1 byte/s y mínimo RTT HTTP <=1 s además de restricciones
de cómputo. Son umbrales de demostración local, no recomendaciones de producción.
El modo sondea una vez en la segunda ronda; no renueva muestras automáticamente
para corridas largas que excedan su vigencia.

Ejecución real CPU/CUDA en artifacts/transfer-scheduling-v1: cuatro procesos worker
completaron cuatro trabajos; dos sondas publicadas. Consulta de SQLite confirmó
los tres requisitos HTTP en ambos trabajos de la ronda 2, ambos completed.
Coordinador reiniciado/reintentado mantuvo IDs y commit sin reservas duplicadas.
Delta L2 embeddings .1838996857; duración 31.368 s. Cada worker ganó seis créditos
por dos trabajos verificados; no por las sondas. Informe TRANSFER_SCHEDULING_SMOKE.json.
Sigue siendo un único equipo/loopback, sin evidencia de ancho de banda multihost.

CI incorpora --transfer-scheduling al smoke instalado de dos workers CPU;
resultado remoto de esta ampliación pendiente.

## Renovación opcional en NodeAgent

`NodeAgent(..., transfer_probe_interval=60)` mide y publica una sonda de 64 KiB
antes de reclamar trabajo, después del heartbeat. Por defecto es `None` y no
produce tráfico adicional. Se aceptan enteros entre 60 y 120 segundos; el plazo
se cuenta con reloj monotónico desde el final del intento anterior, incluso si
falla. No hay hilo de sondeo: solo se comprueba al llamar a `run_once`.

`last_transfer_measurement` conserva el último resultado del intento automático;
`last_transfer_probe_error` contiene únicamente el nombre del tipo de error.
Los errores HTTP o de validación de la sonda no impiden reclamar otros trabajos.
Las muestras siguen siendo declaraciones del cliente y caducan a los 120 segundos.
Un trabajo largo, una llamada tardía o errores pueden dejar un intervalo sin
muestra válida; no se garantiza elegibilidad continua. Las sondas manuales y
otros clientes con la misma credencial comparten el presupuesto del servidor.

Validación local: 30 pruebas de `test_network_probe.py` y
`test_nodes_and_jobs.py` aprobadas, incluida renovación, espera tras HTTP 429,
recuperación, validación del intervalo y ausencia de sondeo por defecto.

## Activación desde escritorio

En DISTRIBUTED COMPUTE, la casilla «Medir y compartir rendimiento HTTP» permite
activar la renovación de 60 segundos al iniciar el nodo. Está desmarcada por
defecto y no se guarda entre sesiones. Para cambiar un nodo activo hay que
 detenerlo y reiniciarlo. La barra de estado indica si se envió la medición o
falló; «enviada» no certifica que siga vigente en el servidor. El hilo recibe un
booleano capturado en la interfaz, sin acceder a variables Tk.

Verificación local: navegación de las diez secciones en Tk real y activación /
desactivación de la casilla; 36 pruebas de escritorio y privacidad aprobadas.
No se ha realizado todavía revisión visual de esta nueva disposición.
