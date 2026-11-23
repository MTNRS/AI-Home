# Estado de implementación

Última actualización: 2026-11-23

## Fase actual

**Fase 2 — API e identidad inicial; seguridad de producción pendiente.**

## Completado

- Repositorio Git inicial conectado a GitHub.
- Principios, límites arquitectónicos y hoja de ruta documentados.
- Definido el enfoque inicial de monolito modular, PostgreSQL autoritativo y
  nodos voluntarios no confiables.
- Implementado un núcleo de libro mayor con asientos balanceados, saldos no
  negativos por cuenta, operaciones idempotentes y reserva/liquidación/
  reembolso componibles.
- Licencia Apache-2.0 añadida al código del repositorio.
- Backend inicial con configuración por entorno, API FastAPI, registro,
  login/logout con sesiones opacas revocables y endpoint de perfil propio.
- Contraseñas con Argon2id; los tokens se almacenan como hashes.
- Migración Alembic inicial para identidad, sesiones y ledger, más Compose de
  desarrollo para PostgreSQL y CI Python 3.11/3.12.
- Pruebas de API para registro, normalización, autenticación, sesión, cierre y
  respuestas sin secretos.
- Suite local: `python -m pytest -q` — 17 pruebas aprobadas.
- Migración Alembic inicial verificada en SQLite, comparada con el metadata
  ORM y probada en downgrade; Uvicorn inició y `/health` respondió HTTP 200.

## Aún no implementado

- Verificación e integración de API y migraciones contra PostgreSQL real.
- Endurecimiento de producción de autenticación: límite de intentos,
  verificación de correo y recuperación de cuenta.
- Scheduler, nodo de cómputo o sandbox.
- Integración del libro mayor con trabajos verificados y autorización.
- Integración de modelos, chat o memoria.
- Rastreo web, RAG, entrenamiento o motor de evolución.
- Pruebas de concurrencia del ledger y de integración PostgreSQL.
- Interfaz web, cliente de escritorio e instalación de producción.
- TLS, rate limiting, CSRF si se añaden cookies, alertas y respuesta a
  incidentes antes de exposición pública.

Docker no está instalado o no está disponible en PATH en el entorno de
desarrollo actual, por lo que no se pudo validar el arranque del contenedor
PostgreSQL.

## Próximos criterios de salida

Antes de activar trabajos o créditos:

1. Probar migración inicial y consultas críticas contra PostgreSQL real.
2. Añadir rate limiting, política de sesión/credenciales y configuración TLS
   antes de cualquier exposición pública.
3. Implementar autorización en cada nuevo recurso de usuario.
4. Probar concurrencia, duplicación y recuperación del ledger.
5. Implementar política de tareas que impida distribuir datos privados.
6. Desarrollar cliente con permisos explícitos y límites locales.

Actualizar este archivo únicamente después de verificar los cambios descritos.
