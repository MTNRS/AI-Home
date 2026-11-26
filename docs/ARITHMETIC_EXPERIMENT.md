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

## Repetición con semillas fijadas 7 y 123

Se añadieron dos ejecuciones tras observar el resultado inicial, sin cambiar
datos, particiones, arquitectura, precisión, LR, pasos ni regla de generación.
Las semillas 7 y 123 se declararon antes de ejecutar esas repeticiones. El CLI
acepta `--seed`; el valor predeterminado sigue siendo 42. No se seleccionó ni
promovió un candidato a partir de estos resultados.

| Semilla | Tras 1 paso | Tras 1.000 pasos |
| --- | --- | --- |
| 42 | 0/14 | 13/14 |
| 7 | 0/14 | 6/14 |
| 123 | 0/14 | 7/14 |

Media descriptiva final: 61,9%; rango 42,9–92,9%. Son los mismos 14 problemas
repetidos, no 42 problemas independientes. Tres semillas son insuficientes para
establecer fiabilidad general. La mejora desde el inicio se repite, pero el
13/14 inicial no representa un resultado estable del procedimiento. Este
resultado señala sensibilidad a inicialización/muestreo; no identifica por sí
solo la causa ni justifica modificar una versión publicada.

Informes completos [semilla 7](ARITHMETIC_SEED7.json) y
[semilla 123](ARITHMETIC_SEED123.json); checkpoints y protocolos locales en
`artifacts/arithmetic-seed7-v1` y `artifacts/arithmetic-seed123-v1`.

## Robustez de formato sin reentrenamiento

Evaluados los tres checkpoints existentes con 14 pares reservados en cuatro
formatos: `a+b=`, `a + b =`, `Suma a+b` y `Add a+b`. Se fijó la suite antes de
inferir; greedy, máximo cuatro tokens, CUDA. Todos los prompts caben en contexto.
Los hashes del checkpoint identifican cada evaluación en el informe completo.

| Semilla | Original | Espacios | Suma | Add |
| --- | --- | --- | --- | --- |
| 42 | 13/14 | 3/14 | 0/14 | 0/14 |
| 7 | 6/14 | 3/14 | 0/14 | 1/14 |
| 123 | 7/14 | 8/14 | 6/14 | 9/14 |

La generalización depende marcadamente del formato y la semilla. Las variantes
con palabras también omiten `=`, por lo que no aíslan causalmente comprensión
del idioma. Sus etiquetas spanish/english identifican la forma de la pregunta,
no una competencia lingüística. Son 14 problemas repetidos, no 56 problemas
independientes por checkpoint. No hubo entrenamiento adicional ni selección de
un ganador. El checkpoint mejor en el formato original no es el más robusto.

Reproducción: `python -m ai_home.foundation.arithmetic_robustness --checkpoints
RUTA1 RUTA2 RUTA3 --output-dir NUEVO --device cuda` (en una sola línea).
[Informe completo](ARITHMETIC_ROBUSTNESS.json). Artefactos locales en
`artifacts/arithmetic-robustness-v1`; script y suite procedural versionados.

## Hipótesis: ampliar formatos de entrenamiento

Tras observar fragilidad, se fijó la siguiente intervención: expandir solamente
los 70 pares train a cuatro formatos (280 registros); conservar las 16 preguntas
de validación originales y los pares de desarrollo reservados. Modelo, semillas
42/7/123, LR, precisión, batch y 1.000 actualizaciones permanecen iguales. El
presupuesto de pasos/contexto coincide, aunque cambian ejemplos muestreados y
longitudes de entrada; no es igualdad de datos vistos ni de tiempo exacto.

Las tres ejecuciones CUDA desde cero finalizan con resultados por formato
(original/espacios/Suma/Add) de 10/14 en semilla 42, 11/14 en 7 y 8/14 en 123.
La invariancia de puntuación mejora, pero no basta para aceptación: la comparación
por probe mediante `evolution.probe_regressions` detecta 4 regresiones en semilla
42, ninguna en 7 y 3 en 123 respecto de sus controles. Se rechaza la promoción
incondicional; no se publica un candidato. El resultado de semilla 7 requiere
confirmación independiente, no selección retrospectiva como ganador.

Esta suite ya había sido inspeccionada y motivó la intervención: es evaluación
de desarrollo, no prueba independiente final. Ningún par reservado ni su
permutación entró en entrenamiento. Dos pruebas del protocolo comprueban la
separación. El campo `selection` del CLI se aclaró a «sin ajustes dentro de la
ejecución»; los protocolos locales anteriores conservan su texto histórico,
que no debe interpretarse como ausencia de adaptación a resultados previos.

Reproducir con `arithmetic_experiment --augment-formats --seed N` y después
`arithmetic_robustness` sobre los tres candidatos. Informes publicados:
[comparación](ARITHMETIC_AUGMENTATION_COMPARISON.json) y
[evaluación completa](ARITHMETIC_AUGMENTATION_EVALUATION.json). Checkpoints,
configuraciones y datos locales en `artifacts/arithmetic-aug42-v1`,
`artifacts/arithmetic-aug7-v1`, `artifacts/arithmetic-aug123-v1` y
`artifacts/arithmetic-aug-robustness-v1`. No se afirma mejora de lenguaje general.
