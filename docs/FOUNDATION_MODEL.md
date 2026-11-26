# AI@Home Foundation Model

Estado: implementación inicial local, mecánica neuronal validada en CPU. No es una entrega completa del objetivo Foundation Model.

El paquete services/api/src/ai_home/foundation separa arquitectura, tokenizador, datos, checkpoints, entrenamiento e inferencia. Todos los parámetros se inicializan aleatoriamente; no importa pesos ni tokenizadores de terceros. El GPT byte-level anterior se conserva.

Arquitectura: decoder causal, RoPE, GQA configurable, RMSNorm, SwiGLU, conexiones residuales, embeddings compartidos opcionales, KV cache y gradient checkpointing. Nano por defecto usa anchura 128, cuatro capas y contexto 256. Es una configuración conservadora de arranque, no una elección óptima demostrada en la RTX 4070 Super. El smoke usa anchura 32 y dos capas para verificar mecánica.

Micro, Small, Medium y Large siguen pendientes de benchmarks y presupuestos medidos. No se atribuye capacidad lingüística, razonamiento o multimodalidad. La estimación de memoria es aproximada y rechaza configuraciones excesivas; todavía no realiza autoajuste mediante benchmarks.

El proveedor de escritorio AI@Home Foundation acepta el nombre de un checkpoint .pt del directorio de modelos de AI@Home. No tiene fallback a un proveedor externo. El proveedor byte-level mantiene su comportamiento previo. El servicio HTTP expone Foundation mediante `/api/v1/chat/completions`, con autenticación y checkpoint configurado por el operador. La prueba de integración entrena un checkpoint y compara las respuestas HTTP con inferencia directa real. La interfaz gráfica sigue sin revisión visual.

Referencias técnicas consultadas: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention y https://docs.pytorch.org/docs/stable/notes/serialization . La implementación utiliza primitivas de PyTorch y no incorpora una implementación completa de otro modelo.

