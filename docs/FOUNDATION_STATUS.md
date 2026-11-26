# Estado del objetivo Foundation — 2026-11-26

Este inventario conserva el alcance del objetivo original. No es una declaración
de aceptación ni una auditoría exhaustiva concluida. La evidencia enumerada cubre
las ejecuciones indicadas; no demuestra capacidades generales, seguridad absoluta,
calidad conversacional ni escalabilidad comunitaria. Los apartados históricos de
otros documentos describen el estado en que se escribieron, no el estado actual.

## Evidencia vigente de ejecución y entrega

- Rama: `codex/foundation-validation`; PR borrador [#1](https://github.com/MTNRS/AI-Home/pull/1).
- Revisión validada remotamente: `a0008bc` (incluye entrenamiento de escritorio con secuencias en disco). FP16/BF16 CUDA se verificaron localmente, no en los runners CPU.
- [Matriz Linux/PostgreSQL](https://github.com/MTNRS/AI-Home/actions/runs/45843869414): ejecución concluida correctamente.
- [Foundation CPU](https://github.com/MTNRS/AI-Home/actions/runs/45843869417): suite, paquete instalado, API, transferencia HTTP, smokes visuales/Tk, nodo de escritorio, entrenamiento con caché y dos rondas distribuidas con reinicio y restricciones de transferencia concluidos correctamente.
- Experimento visual CUDA posterior: tres colores sólidos, 18 imágenes train y seis intensidades reservadas, 0/6 → 6/6 aciertos después de 200 pasos; control blanco 2/6. Evidencia limitada al generador sintético, no visión general. [Protocolo, resultados y límites](VISION_PREFIX.md).
- Suite local completa en `5f082ec`: 387 correctas y 7 PostgreSQL omitidas, en 47,93 segundos. Incluye protección por familias documentales en entrenamiento, conectores, ampliación de vocabulario, protocolo distribuido y evaluación; estos cambios posteriores a `a0008bc` no quedan cubiertos por la evidencia remota anterior. Sustituye la ejecución histórica con tres fallos por memoria. PostgreSQL local separado anterior: 7 correctas.
- Experimento aritmético CUDA: suma de un dígito con pares/permutaciones reservados. Semillas 42/7/123: 13/14, 6/14 y 7/14 tras 1.000 pasos, frente a 0/14 tras uno. Variación elevada; no demuestra razonamiento general ni se promueve un candidato. [Protocolo e informes](ARITHMETIC_EXPERIMENT.md).
- Preparación opcional de secuencias en disco con equivalencia de tensores y reanudación CPU/CUDA; menor memoria observada en el benchmark sintético, sin garantía de límite estricto de RAM. [Medición](SEQUENCE_MEMORY_BENCHMARK.json). Cancelación cooperativa durante preparación/BPE y opción integrada en escritorio.
- Aumento de formatos aritméticos evaluado con tres semillas: 4/0/3 regresiones por caso respecto a los controles, sin promoción incondicional. Es un experimento de desarrollo motivado por resultados ya inspeccionados, no evaluación independiente. [Comparación reproducible](ARITHMETIC_AUGMENTATION_VERIFIED.json).
- Reproducción con wheel instalado y pruebas CPU/CUDA: [alcance y comandos](CLEAN_REPRODUCTION.md). Visor de trabajos revisado visualmente en 860×640 y 640×480: [evidencia y límites](DESKTOP_VISUAL_REVIEW.md).
- `main` permanece en `2ddea7dd3c2a048212ca3e64b68190a26080ed76`. La copia importada tenía historia independiente; se enlazó con el historial remoto mediante un merge sin cambiar el árbol local. No se hizo force-push.
- La PR es entrega revisable de trabajo experimental, no una versión estable ni finalización del objetivo.

## Requisitos y límites de la evidencia

| Área del objetivo | Implementación/evidencia disponible | Trabajo aún pendiente o no probado |
| --- | --- | --- |
| Preservación y auditoría (1–2) | `FOUNDATION_AUDIT.md`, historial upstream conservado, proveedores y GPT anterior presentes | Auditoría final requisito por requisito sobre el estado de entrega |
| Arquitectura propia (3) | `foundation/model.py`; causalidad, RoPE, GQA, gradientes, KV cache, checkpoints CPU/CUDA; `vision.py` añade proyección de parches propia y prefijo visual | Capacidades lingüísticas/razonamiento generales no demostradas; visión limitada a experimento sintético, audio/vídeo pendientes |
| Tokenizador (4) | `tokenizer.py`, `vocabulary.py`; BPE propio Unicode, serialización, ampliación explícita y pruebas | Entrenamiento de vocabulario sobre un corpus autorizado amplio y evaluación de eficiencia multilingüe |
| Escalado (5) | `COMPUTE_BENCHMARKS.md`: Nano/Micro/Small medidos, `resources.py` ajuste heurístico | Medium/Large sin tamaño justificado; memoria real para más contextos/vocabularios; autoajuste basado en benchmarks |
| Preentrenamiento (6) | `training.py`, smokes CPU/CUDA, resume exacto, optimizador/RNG, cancelación y checkpoints | Entrenamiento prolongado con corpus representativo; mejora de capacidades medida |
| Corpus/conectores (7) | `data.py`, `knowledge_ingest.py`, extractor local Wikipedia, conector GitHub público fijado a commit, procedencia/consentimiento y permisos separados | Más conectores de red y diversidad real de fuentes; corpus grande autorizado/redistribuible |
| Contribuciones (8) | `contributions.py`, exportación congelada/revisada, editor de correcciones consentidas con procedencia, SFT y pruebas extremo a extremo | Alternativas y preferencias; defensas contra poisoning más allá de filtros/revisión y contaminación exacta y familias declaradas; revisión visual en otros tamaños/DPI |
| Aprendizaje continuo (9) | `continual.py`, replay, holdout fijo, candidato separado, comparación y rollback en registro | Evidencia de mejora sostenida sin olvido sobre tareas representativas |
| Distribuido (10) | `distributed.py`, dos procesos CPU/CUDA por HTTP, gradientes recomputados, leases renovables y coordinador CLI reanudable y [asignación con restricciones medidas](MEASURED_SCHEDULING.md) | Dos equipos físicos; ancho de banda/latencia de red independientes del servicio y optimización global; reinicio automático/failover del servicio coordinador; grandes modelos/adaptadores/particiones |
| Inferencia (11) | `inference.py`, SSE, controles, cancelación y selección de versiones en chat; inferencia visual local PNG/JPEG con CLI y ventana experimental, streaming y aislamiento de perfil | Rendimiento sostenido/multicliente; integración visual completa con conversación, registro aprobado y evaluación amplia pendientes |
| Evaluación (12) | `evaluation.py`, suites versionadas; recuperación TF-IDF/neural; evaluación visual greedy con control blanco y experimento de colores documentado | Benchmarks amplios/independientes; métricas lingüísticas y humanas representativas; seis casos sintéticos no prueban visión general |
| Evolución (13) | `evolution.py`, control/experimento, reglas por probe, cancelación, resultados reales rechazados | Más familias de hipótesis y experimentos de arquitectura/optimizador; mejora verificada de capacidades |
| Conocimiento actual (14) | Índices TF-IDF/neuronales, referencias, contexto no privilegiado, versiones inmutables y UI | Actualización explícita GitHub inicial; cobertura de otras fuentes pendiente; relevancia neuronal general no demostrada |
| Privacidad (15) | Aislamiento cuenta/memoria, consentimiento, permisos, filtros, borrado y pruebas cruzadas | Auditoría integral de todos los recursos y conectores futuros; no hay borrado garantizado de influencia en pesos ni vault cifrado |
| Créditos (16) | Ledger, reservas, liquidación, reembolso, inferencia/trabajo y concurrencia PostgreSQL | Resistencia a fraude comunitario más allá de recomputación; políticas/operación a escala |
| Interfaz (17) | Diez secciones nombradas y navegación Tk verificada, chat, training/evaluation/evolution, conocimiento, créditos, inspección/registro; ventana visual local con streaming | Revisión visual de diálogos recientes y DPI, integración de métricas completa; no hay evidencia visual integral |
| Pruebas (18) | CPU/CUDA local; Linux 3.11/3.12/PostgreSQL remoto; entrenamientos y cambios reales de pesos | Ampliar al resto de funcionalidades pendientes y mantener evidencia de cada revisión |
| Aceptación (19) | Los 22 criterios tienen componentes y demostraciones locales parciales descritos arriba | No se reduce el objetivo a sus criterios mínimos; falta auditar todos los requisitos explícitos antes de declarar aceptación |
| Entrega (20) | Código, configuraciones, informes y PR publicados; paquete Nano smoke con pesos/corpus propios, manifiesto y verificador en Actions ([detalle](SMOKE_DISTRIBUTION.md)) | Corpus/evaluaciones representativos redistribuibles, reproducción integral de API/GUI/servicios y auditoría final |
| Prioridad y misión (21–22) | Pesos propios desde cero; modelo externo no sustituye automáticamente Foundation | Plataforma generalista/multimodal/comunitaria sigue siendo objetivo de investigación, no capacidad obtenida |

## Próximas líneas de implementación

1. Conectores de conocimiento explícitos y acotados con revisión de derechos,
   privacidad y procedencia; no habilitar entrenamiento por defecto.
2. Medición y selección distribuida por rendimiento/red, recuperación del
   coordinador y validación multihost cuando exista infraestructura disponible.
3. Flujos de corrección/preferencias consentidas y evaluaciones independientes
   capaces de demostrar o rechazar mejoras reales.
4. Consolidación/revisión visual de la interfaz y reproducción de la entrega
   desde un entorno limpio, incluyendo artefactos redistribuibles.
5. Ampliar la vía visual implementada con datos autorizados representativos y
   evaluación independiente. El aprendizaje de colores sintéticos no sustituye
   comprensión visual general; audio/vídeo siguen pendientes.

Cada línea requiere implementación y evidencia. Una limitación de hardware puede
justificar documentación e instrucciones reproducibles para la parte no ejecutable;
no convierte automáticamente esa parte en una capacidad verificada.
