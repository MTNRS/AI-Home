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
