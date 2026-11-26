# Tokenizador propio

ByteBPE aprende fusiones desde documentos de entrenamiento, comenzando con los 256 bytes. No hereda vocabulario de modelos externos. Mantiene seis identificadores estructurales: pad 256, bos 257, eos 258, system 259, user 260, assistant 261. Las cadenas escritas por usuarios no se interpretan como estos identificadores.

El formato JSON incluye versión, tokens especiales y fusiones ordenadas. Cada checkpoint contiene la tabla completa y su SHA-256. La decodificación de texto válido es reversible; durante generación, secuencias UTF-8 inválidas producen un carácter de sustitución explícito. El streaming usa un decodificador incremental.

El entrenador de referencia está limitado a 8 MB; su algoritmo de recuento completo no es adecuado para corpus mundiales. Un vocabulario ampliado constituye una versión nueva: no se permite carga automática con tamaños incompatibles. La migración explícita de embeddings y del optimizador queda pendiente.

Ejecutado: cuatro pruebas unittest con Unicode, código, tokens estructurales, determinismo, serialización, fusiones inválidas, autorización y deduplicación. No se ha entrenado todavía un tokenizador sobre un corpus de producción autorizado.



## Explicit vocabulary expansion

`python -m ai_home.foundation.vocabulary parent.pt authorized.json expanded.pt --vocab-size 1024`
creates a separate candidate with append-only BPE merges. Existing token IDs,
byte pieces, embedding/output rows and all other weights are preserved exactly.
New rows use the model initializer with an explicit seed. The requested size is
a ceiling; insufficient repeated pairs can produce fewer tokens. No new merges
is an error. Input records pass public training authorization and privacy checks;
previous evaluation documents are rejected, and tokenizer corpus hashes join the
training lineage. The parent and existing outputs cannot be overwritten.

This changes text segmentation and output probabilities, even though old IDs
remain valid. Re-tokenize datasets and evaluate again. The artifact is not an
exact-resume checkpoint: optimizer state and prior evaluation results are omitted.
Use training CLI `--initial-checkpoint expanded.pt` (or the Python trainer's
`initial_checkpoint` argument) for fresh optimizer training,
then the normal evaluation and approval process. No model is activated by migration.
The memory budget checks model tensors, not a hard process-memory limit.

## Cancelación cooperativa del BPE

`ByteBPE.train`, `extend` y `encode` aceptan el argumento opcional `cancel`
(evento con `is_set`). Se comprueba antes de documentos y fusiones y dentro del
conteo de pares/fusión en bloques de hasta 4.096 posiciones. Si se solicita
parada, elevan `TokenizationCancelled` sin devolver un tokenizador o secuencia
parcial. No se modifica el tokenizador existente durante `extend`.

El entrenador transmite su evento al aprendizaje BPE y a la codificación de
secuencias; traduce la excepción a `TrainingCancelled`, conservando la salida de
cancelación de CLI. Sin evento, se conserva la vía habitual. Pruebas comprueban
igualdad exacta del vocabulario, ampliación y codificación con evento no activado,
y cancelación dentro de conteo/fusión. No hay garantía de latencia de tiempo real:
la conversión UTF-8, asignaciones nativas y selección del par más frecuente no
son interrumpidas internamente. Otros consumidores deben pasar el evento para
habilitar este comportamiento.
