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
- Exportación básica y eliminación de cuenta con confirmación de contraseña;
  la eliminación invalida sesiones y elimina nodos/trabajos del usuario.
- Endpoint `/health/ready` comprueba conectividad a la base de datos.
- Protocolo inicial de nodos con credencial aleatoria hasheada, informe de
  hardware por GPU y leases persistentes para el ejecutor determinista
  `math.sum.v1`.
- Los trabajos distribuidos requieren consentimiento explícito registrado; el
  scheduler valida el resultado, reintenta leases vencidos y limita intentos.
- Cliente de nodo Python ejecuta solo handlers tipados y permite una iteración
  de polling, sin ejecutar código arbitrario.
- Inventario local verificado: 16 hilos lógicos, 31 GiB de RAM y una NVIDIA
  GeForce RTX 4070 SUPER con 12 GiB de VRAM; la detección usa `nvidia-smi`.
- CI preparada con PostgreSQL 17 y pruebas de migraciones, API de identidad,
  reclamación concurrente de trabajos y concurrencia del ledger.
- Suite local más reciente: `python -m pytest -q` — 36 pruebas aprobadas y 4
  pruebas PostgreSQL omitidas por no haber servidor disponible localmente.
- Migración Alembic inicial verificada en SQLite, comparada con el metadata
  ORM y probada en downgrade; Uvicorn inició y `/health` respondió HTTP 200.
- Compilación local de módulos Python y `git diff --check` aprobados.

## Aún no implementado

- Confirmar en GitHub Actions que pasen la migración, API y pruebas concurrentes
  con PostgreSQL real; no se pudieron ejecutar localmente.
- Endurecimiento de producción de autenticación: límite de intentos,
  verificación de correo y recuperación de cuenta.
- Detección de hardware en otros equipos y sistemas operativos; CUDA es
  opcional y no se presume.
- Aplicación de escritorio con cuenta, elección de servidor, uso de IA y
  aportación de cómputo opcional.
- Ejecutores de embeddings e inferencia, cliente de nodo de larga duración y
  sandbox reforzado.
- Integración del libro mayor con trabajos verificados y autorización.
- Integración de modelos, chat o memoria.
- Rastreo web, RAG, entrenamiento o motor de evolución.
- Pruebas de concurrencia del ledger y de integración PostgreSQL.
- Interfaz web, cliente de escritorio e instalación de producción.
- TLS, rate limiting, CSRF si se añaden cookies, alertas y respuesta a
  incidentes antes de exposición pública.

Docker no está instalado o no está disponible en PATH en el entorno de
desarrollo actual, por lo que las pruebas PostgreSQL quedan pendientes de la
ejecución de GitHub Actions.

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
