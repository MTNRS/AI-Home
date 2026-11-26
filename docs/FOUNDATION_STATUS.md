# Estado del objetivo Foundation — 2026-11-26

Este inventario conserva el alcance del objetivo original. No es una declaración
de aceptación ni una auditoría exhaustiva concluida. La evidencia enumerada cubre
las ejecuciones indicadas; no demuestra capacidades generales, seguridad absoluta,
calidad conversacional ni escalabilidad comunitaria. Los apartados históricos de
otros documentos describen el estado en que se escribieron, no el estado actual.

## Evidencia vigente de ejecución y entrega

- Rama: `codex/foundation-validation`; PR borrador [#1](https://github.com/MTNRS/AI-Home/pull/1).
- Revisión de código validada remotamente: `2e3d4c40f43d9d9e1b3c85910191fc10a19cd710`.
- [Matriz Linux/PostgreSQL](https://github.com/MTNRS/AI-Home/actions/runs/45812648778): Python 3.11 y 3.12, cada uno 267 pruebas correctas y 8 omitidas por CUDA ausente.
- [Foundation CPU](https://github.com/MTNRS/AI-Home/actions/runs/45812648867): 262 correctas, 13 omitidas (CUDA y PostgreSQL opcionales), entrenamiento smoke real y artefacto subido.
- Última suite Windows/CUDA completa anterior al ajuste de sincronización del test: 270 correctas, 5 PostgreSQL omitidas. Después del ajuste: las 3 pruebas de streaming correctas localmente.
- `main` permanece en `2ddea7dd3c2a048212ca3e64b68190a26080ed76`. La copia importada tenía historia independiente; se enlazó con el historial remoto mediante un merge sin cambiar el árbol local. No se hizo force-push.
- La PR es entrega revisable de trabajo experimental, no una versión estable ni finalización del objetivo.

## Requisitos y límites de la evidencia

| Área del objetivo | Implementación/evidencia disponible | Trabajo aún pendiente o no probado |
| --- | --- | --- |
| Preservación y auditoría (1–2) | `FOUNDATION_AUDIT.md`, historial upstream conservado, proveedores y GPT anterior presentes | Auditoría final requisito por requisito sobre el estado de entrega |
| Arquitectura propia (3) | `foundation/model.py`; pruebas de causalidad, RoPE, GQA, gradientes, KV cache y checkpoints CPU/CUDA | No demuestra capacidades lingüísticas, razonamiento o multimodalidad |
| Tokenizador (4) | `tokenizer.py`, `vocabulary.py`; BPE propio Unicode, serialización, ampliación explícita y pruebas | Entrenamiento de vocabulario sobre un corpus autorizado amplio y evaluación de eficiencia multilingüe |
| Escalado (5) | `COMPUTE_BENCHMARKS.md`: Nano/Micro/Small medidos, `resources.py` ajuste heurístico | Medium/Large sin tamaño justificado; memoria real para más contextos/vocabularios; autoajuste basado en benchmarks |
| Preentrenamiento (6) | `training.py`, smokes CPU/CUDA, resume exacto, optimizador/RNG, cancelación y checkpoints | Entrenamiento prolongado con corpus representativo; mejora de capacidades medida |
| Corpus/conectores (7) | `data.py`, `knowledge_ingest.py`, extractor local Wikipedia, conector GitHub público fijado a commit, procedencia/consentimiento y permisos separados | Más conectores de red y diversidad real de fuentes; corpus grande autorizado/redistribuible |
| Contribuciones (8) | `contributions.py`, exportación congelada/revisada, correcciones consentidas con procedencia, SFT y pruebas extremo a extremo | Interfaz de correcciones, alternativas y preferencias; defensas contra poisoning más allá de filtros/revisión y contaminación exacta |
| Aprendizaje continuo (9) | `continual.py`, replay, holdout fijo, candidato separado, comparación y rollback en registro | Evidencia de mejora sostenida sin olvido sobre tareas representativas |
| Distribuido (10) | `distributed.py`, dos procesos CPU/CUDA por HTTP, gradientes recomputados, leases renovables y coordinador CLI reanudable | Dos equipos físicos; selección por rendimiento/ancho de banda/latencia; reinicio automático/failover del servicio coordinador; grandes modelos/adaptadores/particiones |
| Inferencia (11) | `inference.py`, SSE, controles de generación, cancelación, límites y selección de versiones en chat | Rendimiento sostenido/multicliente; no se afirma inferencia multimodal |
| Evaluación (12) | `evaluation.py`, suites versionadas y reportes; `retrieval_evaluation.py` comparación TF-IDF/neural | Benchmarks amplios/independientes; métricas lingüísticas y humanas representativas. Probes iniciales no prueban calidad |
| Evolución (13) | `evolution.py`, control/experimento, reglas por probe, cancelación, resultados reales rechazados | Más familias de hipótesis y experimentos de arquitectura/optimizador; mejora verificada de capacidades |
| Conocimiento actual (14) | Índices TF-IDF/neuronales, referencias, contexto no privilegiado, versiones inmutables y UI | Actualización explícita GitHub inicial; cobertura de otras fuentes pendiente; relevancia neuronal general no demostrada |
| Privacidad (15) | Aislamiento cuenta/memoria, consentimiento, permisos, filtros, borrado y pruebas cruzadas | Auditoría integral de todos los recursos y conectores futuros; no hay borrado garantizado de influencia en pesos ni vault cifrado |
| Créditos (16) | Ledger, reservas, liquidación, reembolso, inferencia/trabajo y concurrencia PostgreSQL | Resistencia a fraude comunitario más allá de recomputación; políticas/operación a escala |
| Interfaz (17) | Chat, training/evaluation/evolution, conocimiento, créditos, inspección y registro con rollback funcionales | Consolidar secciones nombradas; revisión visual de diálogos recientes; no hay evidencia visual integral |
| Pruebas (18) | CPU/CUDA local; Linux 3.11/3.12/PostgreSQL remoto; entrenamientos y cambios reales de pesos | Ampliar al resto de funcionalidades pendientes y mantener evidencia de cada revisión |
| Aceptación (19) | Los 22 criterios tienen componentes y demostraciones locales parciales descritos arriba | No se reduce el objetivo a sus criterios mínimos; falta auditar todos los requisitos explícitos antes de declarar aceptación |
| Entrega (20) | Código, configuraciones, informes y PR publicados; artefacto de smoke en Actions | Paquete versionado de pesos/datasets redistribuibles, instrucciones verificadas en entorno limpio y auditoría final |
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
5. Diseño e implementación multimodal con datos autorizados y pruebas propias;
   no atribuir esta capacidad al Transformer textual actual.

Cada línea requiere implementación y evidencia. Una limitación de hardware puede
justificar documentación e instrucciones reproducibles para la parte no ejecutable;
no convierte automáticamente esa parte en una capacidad verificada.
