# Hoja de ruta

Las fases son objetivos secuenciales; no implican que estén implementadas. Cada
fase debe cerrar con código ejecutable, pruebas pertinentes, documentación y
un estado honesto en [PROGRESS.md](PROGRESS.md). Los 36 criterios de aceptación
detallados se conservan en el plan de implementación.

## Fase 1 — Base ejecutable y preparación comunitaria

- Monorepo, configuración, migraciones, CI, Compose y documentación operativa.
- Revisar secretos, dependencias, licencias y privacidad antes de abrir el
  repositorio; preparar contribuciones y reporte privado de vulnerabilidades.

## Fase 2 — API, identidad y privacidad

- Autenticación, autorización por recurso, sesiones y persistencia PostgreSQL.
- Probar aislamiento, exportación, retención y eliminación de datos personales.

## Fase 3 — Trabajos y nodos voluntarios

- Contratos versionados, hardware por dispositivo, leases, reintentos,
  cancelación y resultados validados.
- Ejecutar únicamente trabajos tipados y permitidos; nunca código arbitrario.
- Aportar cómputo debe ser optativo, separable del uso de IA y revocable.

## Fase 4 — Créditos integrados

- Conectar el ledger transaccional con trabajos aceptados y validados.
- Comprobar concurrencia, idempotencia, reserva, liquidación y reembolso en
  PostgreSQL real antes de activar recompensas.

## Fase 5 — Cliente de escritorio y proveedores BYOK

- Cliente Windows/Linux para instancias comunitarias y autohospedadas.
- Adaptadores directos para OpenAI, Anthropic, Google Gemini y endpoints
  OpenAI-compatibles.
- Guardar las claves solo en el almacén seguro del sistema operativo; no
  degradar a texto plano. Las llamadas directas salen del cliente al proveedor,
  nunca pasan por AI@Home. Explicar el tratamiento del proveedor externo.
- Mantener la contribución de cómputo separada y opcional.

## Fase 6 — Interfaz web e inferencia propia

- Frontend React/TypeScript funcional, conectado a servicios reales.
- Definir contratos de serving local/remoto e integrar el modelo AI@Home solo
  después de entrenarlo y superar gates reproducibles.
- Identificar claramente los servicios o modelos externos; no presentarlos
  como un modelo propio.

## Fase 7 — Multimodalidad y conocimiento privado

- Imagen, audio, documentos, RAG, citas y memoria privada con control de acceso
  y pruebas negativas entre cuentas.
- Aislar artefactos, índices y datos privados; no enviarlos a nodos voluntarios.

## Fase 8 — Conocimiento web y datos autorizados

- Conectores con procedencia, licencias, límites y tratamiento de páginas como
  datos no confiables.
- Conversaciones candidatas al entrenamiento solo con opt-in por conversación,
  apagado por defecto, revisión de derechos, privacidad, secretos y procedencia.
- Permitir consulta y retirada antes de congelar datasets; advertir que no se
  garantiza deshacer influencia en pesos publicados.

## Fase 9 — Modelo fundacional y evaluación

- Investigar y entrenar desde inicialización aleatoria, no contar LoRA ni
  ajuste de un modelo ajeno como modelo fundacional propio.
- Antes de escalar, medir corpus autorizado, tokenizer, arquitectura, hardware,
  consumo, coste y rendimiento con experimentos reproducibles.
- Versionar datasets, semillas, checkpoints y evaluaciones; requerir aprobación
  humana para promoción. Mantener el estado experimental si faltan recursos.

## Fase 10 — Entrenamiento colaborativo e investigación

- Validar colaboración real de nodos y documentar límites de comunicación y
  hardware físico disponible.
- Probar experimentos aislados, medibles y reversibles; ninguna IA aprueba o
  despliega cambios críticos ni amplía permisos.

## Fase 11 — Operación y entrega v1.0

- Instaladores, actualizaciones firmadas, backups/restauración, observabilidad,
  respuesta a incidentes y pruebas end-to-end.
- Publicar solo después de auditorías de seguridad, datos y licencias; no
  declarar misión o estatus legal sin ánimo de lucro sin sustento explícito.

## Foundation Model — implementación inicial (2026-11-26)

Se añade un subsistema propio en `ai_home.foundation` con BPE desde cero, Transformer RoPE/GQA, entrenamiento, checkpoints e inferencia. El modelo byte-level y las funcionalidades anteriores se conservan. La integración de escritorio añade un proveedor independiente para checkpoints propios.

**Estado: parcial y bloqueado para validación neuronal.** Cuatro pruebas sin PyTorch pasan; no se ha completado entrenamiento, generado un checkpoint ni ejecutado inferencia nueva. PyTorch/pytest no están disponibles y su instalación falla. La escritura de ramas en GitHub devuelve 403 y Git local no puede crear `.git/index.lock`; no se han creado commits.

Consultar [auditoría y matriz completa](FOUNDATION_AUDIT.md), [modelo](FOUNDATION_MODEL.md), [entrenamiento](TRAINING.md) y [pruebas](TESTING.md). Aprendizaje continuo, distribución, créditos de entrenamiento, RAG público, evolución y paneles completos siguen pendientes; no se reduce el objetivo ni se declaran demostrados.



## Contribuciones hacia Foundation — 2026-11-26

Conectada exportación administrativa congelada a entrenamiento SFT local real. Verificados consentimiento, hash y revisión; el resultado es candidato de uso restringido, no publicación. Prueba extremo a extremo: crear contribuciones, aprobar, congelar, exportar, entrenar, guardar y recargar. Replay, fine-tuning incremental y promoción siguen pendientes.

## Aprendizaje incremental — 2026-11-26

Implementados candidatos desde checkpoints propios, BPE congelado, replay por lote, holdout fijo explícito, trazabilidad ancestral, bloqueo de contaminación exacta y propagación de restricciones. Experimento CPU real con informe en CONTINUAL_SMOKE_CPU.json. Promoción/rollback centralizado, replay restringido, capacidades lingüísticas y control general del olvido siguen pendientes.

## Evaluación y evolución experimental — 2026-11-26

Implementados probes de generación versionados y ciclo hipótesis/control/experimento/comparación. Ejecución real CPU: dos candidatos, diez probes cada uno, 0 aciertos y propuesta rechazada. No hay mejora de capacidades demostrada ni promoción automática. Ver EVOLUTION.md e informes JSON.

## Primer trabajo Foundation en nodos — 2026-11-26

Integrada tokenización BPE pública en scheduler/NodeAgent con capacidades declaradas, límites e integridad verificada. Dos procesos worker completaron dos shards por HTTP real en loopback. No equivale a entrenamiento neuronal distribuido; tampoco hay créditos ni verificación entre equipos todavía. Ver DISTRIBUTED_TRAINING.md.

## Entrenamiento neuronal con nodos — 2026-11-26

Integrado foundation.gradient.v1 con scheduler/NodeAgent y verificación por recomputación. Dos identidades de nodo aportan gradientes reales sobre shards públicos; agregación por tokens y AdamW producen checkpoints e inferencia. Demostración de dos rondas por HTTP loopback; solo un equipo y CPU. Créditos, multihost y escalabilidad de verificación pendientes. Ver DISTRIBUTED_TRAINING.md.

## Créditos Foundation — 2026-11-26

Reservas/liquidaciones/reembolsos integrados con el ledger en la misma transacción del trabajo. API de saldo y movimientos propios, cancelación, devolución por leases agotados y por eliminación de cuenta. Cuatro trabajos de gradientes reales pagaron 10 créditos a cada uno de dos usuarios en una base de prueba. Cobro de inferencia, interfaz y concurrencia PostgreSQL siguen pendientes. Ver CREDITS.md.


### Public knowledge retrieval increment (2026-11-26)

Added a versioned reviewed-public TF-IDF index with provenance, bounded context
packing, and opt-in retrieval in authenticated Foundation chat. Retrieved text
and personal memories use user-level context, not system instructions. This is
retrieval, not weight training or a demonstrated improvement in model quality.
See [Knowledge retrieval](KNOWLEDGE.md) for commands,
permissions, tests and limitations. Network connectors and desktop references
remain outstanding.
