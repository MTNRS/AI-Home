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

Entrenamiento neuronal y agregación entre nodos, validación de actualizaciones,
recompensas con créditos, scheduling por GPU/VRAM/red/rendimiento, renovación de
leases para entrenamientos largos y prueba entre equipos. Tokenizar datos no
es entrenar pesos. PostgreSQL concurrente sigue pendiente de servidor de prueba;
SQLite no demuestra garantías de concurrencia de producción.
