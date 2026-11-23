# Hoja de ruta

Las fases son objetivos secuenciales; no implican que estén implementadas. Cada
fase debe cerrar con código ejecutable, pruebas pertinentes, documentación y
un estado honesto en [PROGRESS.md](PROGRESS.md).

## Fase 0 — Base y contratos

- Documentar arquitectura, límites de privacidad, amenazas y gobernanza.
- Elegir licencia y política de contribución antes de publicación pública.
- Acordar contratos iniciales de trabajo y contabilidad.

## Fase 1 — API y persistencia

- Crear configuración reproducible de desarrollo y pruebas.
- Implementar API modular, migraciones y comprobaciones de salud.
- Definir identidad y autorización antes de exponer datos de usuario.

## Fase 2 — Trabajos y créditos

- Implementar estados tipados, concesiones con vencimiento y reintentos.
- Añadir libro mayor transaccional, idempotencia, reserva y liquidación.
- Probar duplicados, fallos parciales, concurrencia y auditoría.

## Fase 3 — Nodo voluntario

- Aplicación de escritorio para elegir servidor y separar el uso de IA de la
  aportación de cómputo, que debe ser siempre opcional.
- Detección de hardware con consentimiento y límites locales.
- Ejecución aislada y de larga duración de trabajos públicos no sensibles.
- Validación de resultados y actualizaciones verificables del cliente.

## Fase 4 — Inferencia y conversación

- Integrar un modelo abierto con licencia y procedencia verificadas.
- API de conversación e interfaz básica conectada al cliente de escritorio.
- No anunciar modelos, rendimiento ni capacidades que no se hayan medido.

## Fase 5 — Privacidad y memoria

- Memoria privada por usuario, exportación y eliminación.
- Controles de acceso y pruebas automatizadas de aislamiento entre cuentas.
- Evaluación de registros, cachés, embeddings y retención.

## Fase 6 — Conocimiento recuperable

- Conectores autorizados, registro de procedencia e indexación incremental.
- Recuperación textual/semántica y citas comprobables.
- Política para licencias, datos personales y retirada de contenido.

## Fase 7 — Contribuciones y evaluación

- Feedback voluntario separado de consentimiento para entrenamiento.
- Datasets versionados y control de contaminación.
- Benchmarks reproducibles y revisión de regresiones.

## Fase 8 — Entrenamiento experimental

- Adaptadores y entrenamiento reproducible con datos autorizados.
- Evaluación independiente y publicación de limitaciones.
- No transferir memoria privada a los nodos comunitarios.

## Fase 9 — Investigación y evolución

- Generación de hipótesis y prototipos en entornos aislados.
- Evaluación comparada con una referencia estable.
- Revisión humana y despliegue gradual; reversión probada.

## Fase 10 — Escala y modalidades

- Ampliar a múltiples modelos, modalidades y ejecución federada solo tras
  validar aislamiento, costes, fiabilidad y gobernanza.
- Revisar periódicamente las políticas con participación comunitaria.
