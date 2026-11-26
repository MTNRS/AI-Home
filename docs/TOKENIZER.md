# Tokenizador propio

ByteBPE aprende fusiones desde documentos de entrenamiento, comenzando con los 256 bytes. No hereda vocabulario de modelos externos. Mantiene seis identificadores estructurales: pad 256, bos 257, eos 258, system 259, user 260, assistant 261. Las cadenas escritas por usuarios no se interpretan como estos identificadores.

El formato JSON incluye versión, tokens especiales y fusiones ordenadas. Cada checkpoint contiene la tabla completa y su SHA-256. La decodificación de texto válido es reversible; durante generación, secuencias UTF-8 inválidas producen un carácter de sustitución explícito. El streaming usa un decodificador incremental.

El entrenador de referencia está limitado a 8 MB; su algoritmo de recuento completo no es adecuado para corpus mundiales. Un vocabulario ampliado constituye una versión nueva: no se permite carga automática con tamaños incompatibles. La migración explícita de embeddings y del optimizador queda pendiente.

Ejecutado: cuatro pruebas unittest con Unicode, código, tokens estructurales, determinismo, serialización, fusiones inválidas, autorización y deduplicación. No se ha entrenado todavía un tokenizador sobre un corpus de producción autorizado.

