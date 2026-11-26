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

## Archivos locales e inferencia por CLI

Instalar el extra `.[llm]`, que incluye Pillow >=12.2,<14.0. El módulo
`foundation.vision_images.load_image` limita el archivo a 8 MB y la imagen a
4 millones de píxeles antes de decodificarla. Solo admite PNG/JPEG de un cuadro.
Aplica orientación EXIF, compone alpha sobre blanco y ajusta con interpolación
bilineal y relleno blanco sin deformar la proporción. No aplica conversión ICC.
Devuelve píxeles CPU FP32 en [0,1], hash del archivo y versión del preprocesamiento;
no devuelve EXIF, nombre de archivo ni rutas. No convierte estos datos en un corpus
ni concede permisos de entrenamiento. Referencia API:
[Pillow ImageOps](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html).

```powershell
$env:PYTHONPATH='services/api/src'
python -m ai_home.foundation.vision_inference --checkpoint candidate.pt --image own.png --prompt "Describe" --device cpu --max-new-tokens 32
```

La CLI carga exclusivamente el formato candidato visual y ejecuta generación
real local. Los checkpoints textuales no son compatibles. No ofrece garantías de
calidad visual. El preprocesamiento debe mantenerse igual durante entrenamiento.

Pruebas locales con Pillow 13.0.0: 6 passed in 1.80s para imágenes e inferencia;
tras añadir prueba de CLI con pesos e imagen reales, 4 passed in 2.76s para imágenes.
Incluyen padding, alpha, EXIF, JPEG, límites, corrupción y rechazo de animación.
La instalación local mostró un conflicto con Streamlit 1.38 heredado del entorno
global (requiere Pillow<11); no se alteró esa instalación. AI@Home no depende de
Streamlit. Se recomienda su entorno aislado documentado para reproducción.

## Dataset visual revisado

`foundation.vision_data.load_visual_manifest` valida un JSON UTF-8 local de hasta
2 MB y 2–4096 registros. Formato `ai-home-visual-dataset-v1`, campo `preprocessing`
igual a `rgb-exif-white-letterbox-bilinear-v1`, y `dataset_sha256` calculado mediante
`data.canonical_hash` sobre `{format, preprocessing, records}`. El orden de los
registros forma parte de la versión. Cada registro contiene exactamente:

- `image`: ruta relativa portable dentro del directorio del manifiesto.
- `image_sha256`: hash de los bytes del archivo revisado.
- `caption`: texto revisado NFC, sin espacios exteriores ni CRLF.
- `source`, `url`, `license`, `date`, `language`: procedencia declarada.
- `training_allowed: true`, `privacy_reviewed: true`, `visibility: public`.
- `split`: `train` o `validation`; ambos deben estar presentes.

Se rechazan rutas absolutas/escapes, hashes duplicados incluso entre particiones,
metadatos incompletos y patrones comunes de credenciales/correos en el texto.
No se realiza OCR ni detección automática de personas, datos sensibles o licencias
en las imágenes: la revisión visual y autorización humanas son obligatorias.
Estos flags son declaraciones, no firmas ni pruebas de autorización. Tampoco se
identifican duplicados perceptuales o imágenes reexportadas con bytes distintos.

`load_visual_record` recibe un registro del manifiesto validado y carga sus píxeles
solo cuando se necesitan; vuelve a comprobar ruta y hash sobre los mismos bytes
que decodifica. No carga todo el corpus en RAM ni envía imágenes a nodos. La
contención de rutas no pretende ser un sandbox contra atacantes locales que
cambien el sistema de archivos concurrentemente.

Pruebas locales: `test_foundation_vision_data.py` junto con
`test_foundation_vision_images.py`: **16 passed in 2.76s**. Incluyen pares
procedurales propios, carga, modificación posterior de imagen/caption, permisos,
privacidad, rutas, duplicados y presupuestos. Todavía falta conectar este formato
con entrenamiento reanudable y evaluación visual independiente.

## Lotes de caption SFT

`foundation.vision_batches.VisualCaptionDataset` valida el manifiesto y selecciona
exclusivamente `train` o `validation`. Construye la conversación con la instrucción
configurable (por defecto `Describe the image.`), descripción y EOS. Desplaza los
objetivos un token y enmascara instrucción, marcadores y padding con -100: solo
caption y EOS contribuyen a la pérdida. Reserva el contexto de todos los parches.
Rechaza ejemplos demasiado largos sin truncarlos silenciosamente. El tokenizador
se proporciona explícitamente y debe coincidir con el vocabulario del modelo.

Las secuencias de texto se preparan en memoria; los píxeles se cargan por ejemplo,
verificando de nuevo su hash. Cada elemento devuelve imagen [3,H,W], inputs y
labels de longitud fija, compatibles con DataLoader. La instrucción y la versión
del tokenizador deben registrarse en el futuro checkpoint de entrenamiento.

Validación: 15 pruebas batches/dataset pasaron en 2.95s; tras añadir ejecución CUDA,
4 pruebas batches pasaron en 3.12s. La prueba CPU/CUDA usa PNG procedimental del
manifiesto, DataLoader, forward, cross-entropy, backward y AdamW, y comprueba que
cambian los pesos visuales. Es un paso técnico, sin medición de comprensión visual.
Falta el bucle persistente con validación, métricas y reanudación del optimizador.

## Entrenamiento visual FP32 reanudable

`foundation.vision_training.train_visual` conecta el manifiesto con el modelo
inicializado desde cero, AdamW, acumulación, warmup/coseno, clipping y validación
teacher-forced sobre la partición reservada. Usa `TrainingConfig(objective='sft',
precision='fp32')`; otras precisiones se rechazan por ahora. La estimación de RAM/
VRAM existente se amplía con la proyección visual; sigue siendo una heurística.

Guarda candidatos `visual-step-N.pt` periódicos y finales con `training_state`:
optimizador, generador de selección de ejemplos, paso, historial y protocolo
(configuración completa, instrucción, hash del dataset, dispositivo y PyTorch).
La reanudación exige coincidencia del protocolo, arquitectura y tokenizador.
`stop_after` interrumpe sin cambiar el horizonte original del scheduler. Cancelar
antes del primer paso no produce checkpoint; después de una actualización guarda
el estado al llegar al límite del paso. No interrumpe kernels individuales.
Los archivos de pesos exportados sin `training_state` siguen sin ser reanudables.

El historial registra pérdidas train/validation, tokens supervisados, tiempo del
paso de entrenamiento y su tasa. La validación se excluye de ese tiempo. Menor
pérdida no acredita comprensión visual. Este trainer no usa ni publica nodos,
créditos o promociones automáticas. El directorio de salida debe ser propio de la
corrida: el guardado atómico reemplaza un archivo del mismo nombre si existe.

Evidencia local CPU/CUDA: 14 pruebas entrenamiento/checkpoints pasaron en 4.24s.
Cuatro pasos continuos y dos + reanudación producen pesos y pérdidas train
exactamente iguales en cada dispositivo probado. Tras refactorizar el guardado
por cancelación, 3 pruebas entrenamiento pasaron en 3.79s. También se rechazan
cambios de instrucción y precisiones no soportadas. La igualdad no se promete
entre dispositivos o versiones distintos. Pendientes CLI de entrenamiento,
smoke persistido fuera de pytest, precisión mixta y evaluación de capacidades.
