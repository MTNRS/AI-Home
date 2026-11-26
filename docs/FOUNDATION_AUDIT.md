# Auditoría inicial y alcance pendiente

Origen: MTNRS/AI-Home, árbol de main 2ddea7dd3c2a048212ca3e64b68190a26080ed76, recuperado mediante conector porque git clone falla por resolución de red. Se materializaron 57 archivos originales. No había modificaciones locales previas. No se dispone de un clon con historial: la copia local es una instantánea.

Revisión inicial de código, no certificación de funcionamiento:

| Componente | Hallazgo | Acción |
|---|---|---|
| local_llm/train_llm | GPT byte-level, posiciones aprendidas, entrenamiento AdamW, inferencia real; sin validación ni resume de optimizador | Conservado; paquete foundation separado |
| conversation_corpus/wikipedia_corpus | Importación con atribuciones y manifiestos | Conservada; adaptación al nuevo esquema pendiente |
| api/models | Consentimiento, revisión y congelación de datasets | Conservados; conexión SFT pendiente |
| jobs/node_worker | Solo math.sum.v1, leases y handlers permitidos | Conservados; entrenamiento distribuido pendiente |
| credits | Ledger transaccional, balanceado e idempotente | Conservado; recompensa por entrenamiento pendiente |
| memory | Vault local por perfil del SO, recuperación léxica | No equivale a aislamiento por cuenta; revisión de cambio de cuenta pendiente |
| providers/desktop | Proveedor byte-level local y externos independientes | Añadida selección Foundation sin fallback externo |

La revisión no cubre todavía cada ruta ni cada riesgo del repositorio. Hay pruebas originales de autenticación, chat, configuración, créditos, escritorio, modelo local, memoria, migraciones, nodos, PostgreSQL, proveedores y contribuciones. No ejecutadas en este entorno por falta de dependencias.

## Matriz de aceptación

1 Tokenizador desde cero: demostrado sobre fixtures, no corpus productivo.
2 Arquitectura propia: escrita, ejecución neuronal pendiente.
3 Inicialización desde cero: escrita, ejecución pendiente.
4 CUDA: código y test condicionado, ejecución pendiente.
5 Checkpoints: código escrito, artefacto real pendiente.
6 Resume: código/test escritos, ejecución pendiente.
7 Inferencia real: código escrito, ejecución pendiente.
8 Integración: adaptador de escritorio escrito, verificación pendiente.
9 Chat: selección escrita, prueba extremo a extremo pendiente.
10 Datasets autorizados: esquema, controles y tests básicos; corpus productivo pendiente.
11 Evaluación: pérdida implementada; benchmarks de capacidades pendientes.
12 Candidatos: metadatos implementados; registro central pendiente.
13 Aprendizaje continuo: pendiente.
14 Contribuciones a entrenamiento: pendiente.
15 Trabajos distribuidos compatibles: pendiente.
16 Colaboración real entre nodos: pendiente.
17 Créditos por entrenamiento: pendiente.
18 RAG público con referencias: pendiente; memoria léxica existente no lo sustituye.
19 Experimento de automejora: pendiente.
20 Aislamiento privado: código existente preservado, nuevas pruebas de ataque pendientes.
21 Pruebas esenciales: solo cuatro pruebas sin torch pasan; resto pendiente.
22 Reproducción: comandos y workflow preparados; ejecución completa bloqueada.

Interfaz de control completa, multimodalidad, embeddings propios, conectores extensibles, replay/SFT/preferencias, protección avanzada contra poisoning, scheduler por recursos y promoción/rollback permanecen dentro del alcance, aún sin implementar. No se declara terminado el objetivo.


## Actualización verificada 2026-11-26

La matriz anterior describe la auditoría inicial. La ejecución actual demuestra en CPU los criterios mecánicos 2, 3, 5, 6 y 7; la suite obtiene 127 aprobadas y 5 omitidas. El criterio 21 sigue parcial por omisiones CUDA/PostgreSQL y componentes futuros. Véase TESTING.md y FOUNDATION_SMOKE_CPU.json. Los criterios restantes no se dan por completados.

## Criterio CUDA verificado

El criterio mecánico 4 (entrenamiento GPU) tiene ahora evidencia directa: suite CUDA y smoke real en RTX 4070 SUPER. Véase CUDA_VALIDATION.md. La auditoría global sigue incompleta: nodos, créditos de entrenamiento, interfaz completa y otros requisitos no quedan cubiertos por estas pruebas.
