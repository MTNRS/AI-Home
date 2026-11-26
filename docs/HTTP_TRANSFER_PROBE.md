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
de ejecutar/verificar trabajos neuronales. Incluyen autenticación, servidor,+HTTP y transporte: **no son ancho de banda físico ni RTT exclusivamente de red**.
No se restan tiempos para inventar una medida de enlace. El payload repetitivo es
conocido y esta sonda no acredita trabajo útil ni protege contra nodos que mientan.

Validación inicial: `test_network_probe.py`, 2 passed in 2.64s. TestClient verifica
flujo real del endpoint, credenciales, integridad, tamaños y rechazo de codificación.
Estas pruebas en proceso no son un benchmark de red. Pendiente: ejecución HTTP
real, integración explícita de nodo, limitación de frecuencia/persistencia y uso
del scheduler con límites y caducidad; medición entre equipos físicos pendiente.
