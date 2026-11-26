# Verificación Foundation — 2026-11-26

## Ejecución actual

Python 3.11.9 y PyTorch 2.15.0+cpu en Windows.

- `python -m pytest -q`: 127 aprobadas, 5 omitidas, 27,27 segundos. Omisiones: una prueba CUDA y cuatro PostgreSQL.
- `$env:PYTHONPATH="services/api/src"; python -m ai_home.foundation.smoke --output-dir artifacts/foundation-smoke-cpu --device cpu`: exit 0 tras corregir impresión Unicode incompatible con consola cp1252.
- El smoke entrena, reanuda, guarda, carga y genera con pesos reales. No usa modelos simulados ni pesos externos.
- Las pruebas comparan exactamente los pesos de entrenamiento continuo y reanudado en CPU, causalidad, cache, gradientes y gradient checkpointing.

Informe persistente: [FOUNDATION_SMOKE_CPU.json](FOUNDATION_SMOKE_CPU.json). Checkpoints y corpus local: `artifacts/foundation-smoke-cpu/`. El informe incluye hash del checkpoint, versiones, métricas, texto generado y dispositivo de inferencia.

La salida generada es incoherente. Este experimento demuestra ejecución, no calidad conversacional. CUDA no está disponible en esta distribución de PyTorch. No se ejecutaron PostgreSQL, interfaz gráfica, colaboración entre nodos ni CI remota.

## Antecedentes

La sesión anterior no disponía de torch/pytest y no pudo ejecutar la validación neuronal. Esa limitación queda superada para CPU. El workflow `.github/workflows/foundation.yml` sigue pendiente de publicación y ejecución remotas.

## Integración HTTP posterior

`python -m pytest -q`: 129 aprobadas, 5 omitidas, 11,59 segundos. `git diff --check`: exit 0. El test de chat entrena pesos reales, verifica autenticación, compara HTTP con inferencia directa en dos conversaciones y comprueba errores con checkpoints corruptos/ausentes sin fallback externo. La generación neuronal no se simula.

## Contribuciones y SFT

Suite completa tras integrar exportaciones revisadas: `python -m pytest -q` — 131 aprobadas, 5 omitidas, 12,65 segundos. `git diff --check`: exit 0. La prueba usa API local real y entrenamiento PyTorch CPU; no invoca proveedores externos. Se comprueban máscara de pérdida, colas cortas, bloqueo de exportación a usuario ordinario, hash y revisión, cambio de parámetros y recarga del candidato. CUDA y PostgreSQL siguen omitidos.

## Aprendizaje incremental con replay

`python -m pytest -q`: 132 aprobadas, 5 omitidas, 12,74 segundos. `git diff --check`: exit 0. La nueva prueba entrena padre y candidato, verifica tokenizador invariable, restricciones heredadas, padre intacto, reanudación bit a bit con replay y bloqueo de contaminación ancestral. Experimento persistente: `python -m ai_home.foundation.continual_smoke --output-dir artifacts/continual-smoke-v1 --device cpu`, exit 0; informe conservado en CONTINUAL_SMOKE_CPU.json.

## Evaluación de generación y evolución

`python -m pytest -q`: 135 aprobadas, 5 omitidas, 13,70 segundos. `git diff --check`: exit 0. Las pruebas entrenan dos candidatos reales, verifican generación determinista, puntuación exacta, prompts truncados, contaminación conocida y requisitos de aceptación. Pruebas unitarias de decisión usan cifras sintéticas únicamente para validar reglas; el experimento persistente usa pérdidas y generaciones reales. Comandos y artefactos en EVOLUTION.md.

## Validación CUDA posterior

Entorno `.venv-cuda` con PyTorch 2.14.1+cu130: `python -m pytest -q` obtiene 139 aprobadas, 4 omitidas, 17,36 segundos. Las omisiones CUDA de entradas anteriores corresponden a la distribución CPU; ahora se ejecutaron todas las pruebas CUDA, incluidas FP32/FP16/BF16, gradientes, caché y reanudación. Ver CUDA_VALIDATION.md para comandos exactos y evidencia persistente.

## Nodos Foundation

Suite completa en entorno CUDA: 146 aprobadas, 4 omitidas (PostgreSQL), 18,63 segundos. Tras endurecer booleanos de consentimiento: 7 pruebas Foundation nodes aprobadas en 1,21 segundos. Se verifican dos nodos, compatibilidad/RAM, idempotencia, manipulación de resultados, aislamiento de listado, rechazo de datos restringidos/secretos y límites de expansión BPE. Smoke adicional real: dos procesos worker, HTTP loopback, 2 shards y 56 tokens; reporte en NODE_TOKENIZATION_SMOKE.json.

## Gradientes neuronales entre nodos

Suite completa en entorno CUDA: 150 aprobadas, 4 omitidas (PostgreSQL), 14,76 segundos. Tras normalizar configuración y eliminar dependencia NumPy del codec: 4 pruebas distribuidas aprobadas, 3,99 segundos. Se verifica equivalencia matemática con actualización monolítica, rechazo de bases antiguas, documentos solapados, artefactos alterados, opt-in y runtime incompatible, y desactivación de nuevas asignaciones. Smoke HTTP real: dos rondas y cuatro trabajos neuronales aceptados; checkpoint recargado e inferencia real. Informe en DISTRIBUTED_GRADIENT_SMOKE.json.

## Créditos de trabajos Foundation

Suite completa final en entorno CUDA: 158 aprobadas, 4 omitidas (PostgreSQL), 37,55 segundos. Migración 0006 probada con upgrade, comprobación de esquema y downgrade sobre SQLite temporal. Pruebas de pagos y asignaciones incluyen saldo insuficiente, conservación, duplicados, fallo inyectado tras asiento, cancelación, leases agotados, eliminación de cuenta, autofinanciación sin emisión y gradientes reales. Smoke financiado por HTTP loopback: 20 créditos sintéticos iniciales, cuatro trabajos verificados y 10 créditos por contribuyente; informe en DISTRIBUTED_FUNDED_SMOKE.json. `git diff --check`: exit 0. Concurrencia PostgreSQL pendiente.
