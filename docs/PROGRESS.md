# Estado de implementación

Última actualización: 2026-11-23

## Fase actual

**Base (fase 0) completada; preparación del backend y contabilidad.**

## Completado

- Repositorio Git inicial conectado a GitHub.
- Principios, límites arquitectónicos y hoja de ruta documentados.
- Definido el enfoque inicial de monolito modular, PostgreSQL autoritativo y
  nodos voluntarios no confiables.
- Implementado un núcleo de libro mayor con asientos balanceados, saldos no
  negativos por cuenta, operaciones idempotentes y reserva/liquidación/
  reembolso componibles.
- Añadidas pruebas del núcleo con SQLite en memoria; no se ha validado aún
  contra PostgreSQL.
- Validación local: `python -m pytest` — 6 pruebas aprobadas.

## Aún no implementado

- API o backend ejecutable.
- Registro, autenticación o autorización.
- Scheduler, nodo de cómputo o sandbox.
- Integración del libro mayor con trabajos verificados y autorización.
- Integración de modelos, chat o memoria.
- Rastreo web, RAG, entrenamiento o motor de evolución.
- API/integration tests, migraciones y despliegue (por ahora solo hay pruebas
  unitarias del dominio de créditos).

## Próximos criterios de salida

Antes de activar trabajos o créditos:

1. API y esquema de base de datos reproducibles.
2. Autorización probada en cada recurso de usuario.
3. Migraciones para el libro mayor y validación transaccional contra PostgreSQL.
4. Pruebas de concurrencia, duplicación y recuperación ante fallos.
5. Política de tareas que impida distribuir datos privados.
6. Cliente con permisos explícitos y límites locales.

Actualizar este archivo únicamente después de verificar los cambios descritos.
