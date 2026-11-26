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
