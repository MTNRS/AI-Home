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

## Ensayo reproducible persistido

```powershell
python -m ai_home.foundation.vision_smoke --output-dir artifacts/visual-new --device cpu
python -m ai_home.foundation.vision_smoke --output-dir artifacts/visual-cuda-new --device cuda
```

Cada comando exige un directorio nuevo. Conserva cuatro PNG procedurales propios,
manifest JSON, candidatos continuos/interrumpidos/reanudados y report.json. BPE se
entrena solo con captions de entrenamiento. El ensayo verifica cambios en pesos
visuales, reanudación exacta y generación desde checkpoint vuelto a cargar.

Ejecuciones locales reales: CPU `artifacts/vision-smoke-cpu-v1`, CUDA
`artifacts/vision-smoke-cuda-v2`. Informes versionados: VISUAL_SMOKE_CPU.json y
VISUAL_SMOKE_CUDA.json. Modelo 28.544 parámetros, cuatro pasos, máximo cambio
visual 0.0005627163, reanudación exacta en ambos dispositivos. La salida fue
`KKKKKKKK`; no hay descripción correcta ni comprensión visual demostrada.
La segunda ejecución CUDA sincroniza antes/después del paso para medir su tiempo;
la primera, anterior al ajuste, se conserva localmente pero no es el informe
versionado. Las tasas de esta prueba diminuta no comparan rendimiento de hardware.

CI incorpora el ensayo CPU con paquete instalado y Python -I, y publica sus
artefactos; su resultado remoto debe verificarse después de este commit.

## Verificación remota y carga entre plataformas

Commit de código `2f382a4`, [Foundation CI 45830549581](https://github.com/MTNRS/AI-Home/actions/runs/45830549581):
terminado correctamente. Suite **329 passed, 18 skipped**; ensayo visual del
paquete instalado con Python -I completado, exact_resume=true, delta visual
0.0005627162754535675, salida KKKKKKKK. También pasaron empaquetado textual, Tk,
API y dos nodos CPU con reinicio de coordinador. Artefacto `visual-smoke` publicado.

[Tests 45830549637](https://github.com/MTNRS/AI-Home/actions/runs/45830549637):
Python 3.11 y 3.12 con PostgreSQL, **336 passed, 11 skipped** en cada versión.
Los skips corresponden a condiciones opcionales; este runner no prueba CUDA.

El artefacto Linux se descargó a `artifacts/visual-linux-ci-v1` y su checkpoint
`resumed/visual-step-4.pt` se cargó con el validador real en Windows:
SHA256 `bfb455e3a035e85aec507e2228626e8d2e120fe954569fc529460493cdb2a811`,
paso 4, 28.544 parámetros. Esto verifica carga de pesos entre estos entornos,
no reanudación bit-exact entre sistemas ni calidad visual.

## CLI de entrenamiento visual

```powershell
python -m ai_home.foundation.vision_train_cli --dataset artifacts/vision-smoke-cpu-v1/dataset.json --config configs/foundation/vision-caption-example.json --output-dir artifacts/visual-partial-new --stop-after 2
python -m ai_home.foundation.vision_train_cli --dataset artifacts/vision-smoke-cpu-v1/dataset.json --config configs/foundation/vision-caption-example.json --output-dir artifacts/visual-resumed-new --resume artifacts/visual-partial-new/visual-step-2.pt
```

Configuración explícita JSON limitada a 64 KiB: model (sin vocab_size), vision,
training, prompt y tokenizer_vocab_size. En una corrida nueva BPE se entrena solo
con captions train; el tamaño real puede ser inferior al máximo solicitado si no
hay merges repetidos. Al reanudar se utiliza el tokenizador guardado, sin ampliarlo.
El directorio debe ser nuevo para evitar sobrescrituras. Ctrl+C solicita guardado
al límite del paso y restaura después el manejador de señal anterior.

`training-report.json` registra configuración efectiva, hashes, historial y nombre
del candidato. No acredita calidad. Ejecución local real en procesos separados:
`artifacts/visual-cli-partial-v1` paso 2 y `artifacts/visual-cli-resumed-v1` paso 4.
Prueba CLI: 1 passed in 3.10s, entrenamiento real, BPE solo train y salida exclusiva.
El ejemplo es para comprobación técnica diminuta, no un corpus de capacidad.

## Evaluación visual por generación libre

```powershell
python -m ai_home.foundation.vision_evaluation --checkpoint artifacts/visual-cli-resumed-v1/visual-step-4.pt --dataset artifacts/vision-smoke-cpu-v1/dataset.json --prompt "Describe the image." --max-new-tokens 16 --output artifacts/visual-evaluation-new.json
```

Evalúa solo registros marcados validation, con greedy decoding y comparación
exacta después de NFC, normalización CRLF y strip. Repite cada caso con píxeles
blancos como control. Guarda respuestas, aciertos, cambios de respuesta, tokens,
agotamiento del presupuesto, latencia sincronizada CUDA y hashes de protocolo,
dataset, checkpoint y tokenizador. Rechaza prompts que ya exceden el contexto.
No modifica pesos ni promueve candidatos. Exige un archivo de salida nuevo.

Resultado real del candidato CLI de cuatro pasos: **0/1 aciertos**, control blanco
0/1, respuestas cambiadas 0. Ambas salidas K repetida 16 veces agotaron el
presupuesto. Informe: VISUAL_EVALUATION_INITIAL.json. El único caso amarillo es
una comprobación negativa diminuta, no un benchmark de capacidad general. Una
coincidencia exacta puede penalizar descripciones válidas alternativas; diferencias
con el control tampoco prueban comprensión. La partición declarada no demuestra
por sí sola ausencia de contaminación en otros entrenamientos o datos previos.

Prueba del evaluador con modelo real: 1 passed in 3.36s, selección de holdout,
control, determinismo de respuesta/protocolo y rechazo de prompt excesivo.

## Experimento controlado: tres colores sintéticos

```powershell
python -m ai_home.foundation.vision_color_experiment --output-dir artifacts/color-new --device cuda
```

Protocolo fijado antes de ejecutar: 24 PNG propios sólidos RGB de 8×8, tres
categorías (red/green/blue), 18 ejemplos train con intensidades 140/160/180/200/220/240
y seis validation con 150/210. Canales no dominantes a cero. BPE solo train,
28.768 parámetros, inicialización aleatoria seed 42, 200 pasos AdamW FP32,
batch 4, LR .003, warmup 10, sin selección de checkpoint por validación. Instrucción
`Color?`, greedy, máximo 8 tokens, comparación exacta. No se ajustó el protocolo
tras observar el resultado. La validación se calcula durante el entrenamiento;
este conjunto no es un test externo secreto ni representa imágenes naturales.

Ejecución CUDA local `artifacts/visual-color-experiment-v1`: **0/6 antes, 6/6 después**.
Las seis respuestas correctas terminaron sin agotar el presupuesto. El control
blanco siempre respondió green, acertando 2/6; cuatro respuestas cambiaron al usar
la imagen correcta. Pérdida train inicial 5.626584, final .020696; validation final
.020262. Tiempo sumado de pasos de entrenamiento sincronizados: 3.4933 s, excluye
validación/guardado/evaluación y no es tiempo total de ejecución.

Informe completo con hashes y respuestas: VISUAL_COLOR_EXPERIMENT.json. Se
conservan imágenes, manifiesto, checkpoint inicial y candidatos pasos 100/200.
El resultado demuestra aprendizaje neuronal de esta tarea diminuta y evidencia de
condicionamiento visual. No acredita objetos, escenas, fotografías, razonamiento,
robustez a colores mixtos ni generalización fuera de este generador. No se promovió
el candidato ni se sustituyó ningún modelo activo.

## Ventana de inferencia visual local

En FOUNDATION MODEL → «Probar imagen local (experimental)» se seleccionan
checkpoint visual, PNG/JPEG, instrucción y CPU/CUDA. Generar ejecuta en el worker
existente; Cancelar solicita parada entre tokens. Solo hay una ventana visual por
instancia. Cerrar la ventana o cambiar de perfil la destruye y solicita cancelación;
los callbacks usan la protección de sesión existente. No se persisten imagen,
instrucción o respuesta en conversaciones ni se envían al servidor o entrenamiento.
El cargador limita pesos e imágenes igual que la CLI. No hay preview de imagen ni
streaming incremental en esta primera ventana: la respuesta se muestra al terminar.

Prueba funcional real Windows Tk, `artifacts/verify-visual-dialog.py`: abrió el
formulario, seleccionó checkpoint propio del experimento, PNG red-150 e instrucción
Color?, invocó Generar y comprobó respuesta red; cerró las ventanas. La prueba usó
la operación real con ejecución síncrona del harness; el código de producto usa
_background. También pasó scripts/verify_desktop_sections.py con las diez rutas.
No es una revisión visual de layout, DPI o accesibilidad; estas siguen pendientes.

## Streaming de la ventana visual

La ventana muestra fragmentos incrementalmente mediante una cola de 16 elementos,
consumida en el hilo Tk cada 40 ms. Si la cola se llena, el worker espera con
comprobación de cancelación; no acumula eventos sin límite. La respuesta completa
reemplaza el texto al finalizar, evitando duplicar fragmentos pendientes. El poll
verifica ventana y token de sesión; al cerrar/cambiar perfil solicita cancelación.
La cancelación previa evita incluso cargar archivos. Sigue sin persistir historial.

Pruebas: test_visual_desktop.py, 2 passed in 4.22s (stream neuronal real frente a
respuesta final y salida de productor con cola llena al cancelar). Se repitieron
el formulario Tk con generación red y la navegación de diez secciones, ambos OK.
La inspección visual/DPI y una prueba de cambio de perfil durante generación siguen
pendientes; las comprobaciones funcionales anteriores no equivalen a esa revisión.

## Aislamiento visual durante cambio de perfil

`scripts/verify_visual_profile.py` ejecuta Tk real, _background y la función real
_switch_memory_profile con cuentas sintéticas sin conexión. Un productor controlado
emite un fragmento y espera cancelación; tras cambiar perfil, confirma destrucción
de la ventana, parada del productor y descarte del resultado tardío por la guarda
de sesión. La ventana nueva permanece vacía. Es una prueba de concurrencia/privacidad,
no una simulación de capacidad del modelo.

Ejecución Windows pasó. Primer intento del harness usó un identificador de usuario
no UUID y falló antes del cambio; se corrigió a UUID sintético válido y se repitió
correctamente. CI añade la misma prueba sobre paquete instalado con Xvfb/Python -I;
resultado remoto de esta adición todavía pendiente. Revisión visual/DPI pendiente.
