# Entrada visual experimental desde cero

`foundation.vision.VisionPrefixModel` combina una proyección lineal propia de
parches RGB con el decoder Foundation existente. Todos los pesos se inicializan
desde cero. Una imagen de tamaño fijo se divide en parches sin solapamiento,
ordenados por filas, y se antepone al texto. RoPE posiciona la secuencia completa;
la atención sigue siendo causal también entre parches. No hay encoder preentrenado.

El llamante proporciona tensores RGB finitos en [0, 1], con tamaño y dispositivo
explícitos. Los parches consumen el presupuesto de contexto. Los logits devueltos
corresponden solo a las posiciones de texto y pueden entrenarse con objetivos del
siguiente token. La primera llamada requiere imagen; las siguientes pueden usar
KV cache sin repetirla. No se implementan todavía imágenes múltiples ni resolución
variable. El modelo textual conserva sus claves de estado y su API anterior.

Validación local Windows, PyTorch CUDA, 2026-11-26:

```powershell
.venv-cuda\Scripts\python.exe -m pytest -q services/api/tests/test_foundation_vision.py services/api/tests/test_foundation_model.py --tb=short
```

Resultado: **19 passed in 17.52s**, incluidas las variantes CPU y CUDA. Se verifica
condicionamiento de logits por píxeles, causalidad textual, equivalencia KV cache,
gradientes visuales/textuales, actualización AdamW real con gradient checkpointing,
límites de píxeles/contexto, reproducción exacta mediante state_dict y equivalencia
de la ruta textual. Las imágenes son tensores procedurales de prueba sin datos
personales. No constituyen una evaluación de comprensión visual.

Los pesos visuales se guardan con `save_vision_weights` y se cargan con
`load_vision_weights` de `foundation.vision_checkpoint`. El formato independiente
`ai-home-vision-weights-v1` conserva configuración visual/textual, BPE y su hash,
pesos FP32 y procedencia declarada del dataset (SHA256, visibilidad, permiso de
entrenamiento y revisión de privacidad). Es un candidato local: esas declaraciones
no son una certificación de licencia o consentimiento. El guardado atómico puede
reemplazar el destino local indicado; no publica ni promueve el modelo.

La carga usa `weights_only=True`, presupuesto de archivo/modelo de 64 MB por
defecto, hash del mismo descriptor antes/después, construcción meta previa a la
reserva de parámetros, validación de claves/formas/finitud y coherencia de pesos
compartidos. El SHA256 real queda en `model.checkpoint_sha256`. El formato textual
rechaza estos archivos. No se incluye estado del optimizador ni reanudación.

Prueba adicional: `test_foundation_vision_checkpoint.py` entrena un paso real,
guarda/carga y reproduce exactamente los logits; rechaza nueve alteraciones de
pesos/metadatos, presupuestos insuficientes y guardado sin procedencia.

La inferencia local `foundation.vision_inference.stream_vision_tokens` y
`stream_vision_text` reutiliza el muestreador textual (temperatura, top-k, top-p,
penalización de repetición, semilla, EOS, cancelación y decodificación UTF-8).
Recibe una imagen RGB normalizada, ya situada en el dispositivo del modelo.
Conserva una copia de esa imagen durante la sesión. Cuando se llena el KV cache,
reconstruye el contexto con todos los parches y el texto más reciente que quepa;
no descarta silenciosamente el condicionamiento visual. El contexto textual útil
es el contexto total menos el número de parches.

Validación de generación: 18 pruebas pasaron en 18.23s entre
`test_foundation_vision_inference.py` y `test_foundation_model.py`, verificando
generación real, reconstrucción del contexto, uso del cache, reproducibilidad de
muestreo, cancelación y regresión textual. No mide calidad visual.

Pendiente: carga y preprocesamiento de imágenes, datasets autorizados,
entrenamiento reproducible con
reanudación, evaluación independiente e interfaz. El
formato actual de checkpoint textual no acepta este wrapper. No se publica ni
activa automáticamente, y no se conecta a contribuciones o nodos voluntarios.
No se ha demostrado capacidad de describir, reconocer o razonar sobre imágenes.
