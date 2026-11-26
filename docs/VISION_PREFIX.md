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

Pendiente: formato seguro de checkpoint multimodal con procedencia, carga y
preprocesamiento de imágenes, datasets autorizados, entrenamiento reproducible con
reanudación, inferencia de alto nivel, evaluación independiente e interfaz. El
formato actual de checkpoint textual no acepta este wrapper. No se publica ni
activa automáticamente, y no se conecta a contribuciones o nodos voluntarios.
No se ha demostrado capacidad de describir, reconocer o razonar sobre imágenes.
