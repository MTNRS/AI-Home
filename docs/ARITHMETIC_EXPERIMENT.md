# Experimento de suma de un dígito

Protocolo v1 fijado antes de entrenar: modelo propio de 90.816 parámetros,
inicialización semilla 42, BPE byte-level sin merges (262 tokens), SFT,
1.000 pasos AdamW, batch 16, LR .002 con warmup 30 y decay; FP32 CUDA.
Datos procedurales propios Apache-2.0: los 100 pares ordenados de dígitos 0–9.
La partición depende del par **no ordenado** módulo 7: 70 train, 16 validación,
14 prueba. Un par y su permutación permanecen juntos. No se utiliza prueba para
elegir pasos, configuración o vocabulario. La comparación inicial es después
de una actualización, no pesos completamente aleatorios.

Resultado de la primera ejecución: 0/14 → 13/14 respuestas exactas greedy,
con límite de cuatro tokens y todos los prompts dentro de contexto. El único
fallo final es `0+0=` → `1` en lugar de `0`. El cero como respuesta no aparece
en entrenamiento porque su único par pertenece a prueba. Pérdida final train
0.0033221, validación 0.2738689; cambio L2 de embeddings desde la reanudación
12.7333. Son datos sintéticos y una sola semilla; no prueban comprensión de
instrucciones, lenguaje natural, suma de varios dígitos ni razonamiento general.
No se promovió el candidato ni se ajustó el protocolo tras ver el resultado.

Reproducir con entorno llm/CUDA instalado:

```powershell
python -m ai_home.foundation.arithmetic_experiment --output-dir artifacts/arithmetic-new --device cuda
```

El directorio debe ser nuevo. Conserva protocolo, particiones, suite, checkpoints
con optimizador y evaluaciones completas. CPU también está admitida por el script,
pero esta ejecución del experimento fue CUDA. Informe publicado:
[ARITHMETIC_EXPERIMENT.json](ARITHMETIC_EXPERIMENT.json). Artefactos locales en
`artifacts/arithmetic-experiment-v1`. La prueba del protocolo valida resultados,
55 grupos no ordenados, separación de permutaciones y pertenencia de los probes.

Antes de este experimento, suite local completa: 363 aprobadas, 7 PostgreSQL
omitidas, 44.13 segundos. La nueva prueba del protocolo pasó por separado.
