# AI@Home

AI@Home es un proyecto comunitario de inteligencia artificial abierta, con el
objetivo de permitir investigación colaborativa y computación voluntaria sin
comprometer la privacidad de sus participantes.

El proyecto está en desarrollo temprano. Ya incluye una API inicial de
identidad y sesiones, migraciones de base de datos y un núcleo interno de libro
mayor. Todavía no ofrece un nodo de cómputo ni un flujo de créditos conectado
a trabajos validados.

## Principios

- **Consentimiento:** ningún dispositivo, dato o contribución se utiliza sin
  autorización explícita y revocable.
- **Privacidad por diseño:** conversaciones y memorias privadas se mantienen
  separadas de los datos públicos y de las contribuciones globales.
- **Evolución verificable:** una propuesta generada por IA no se considera una
  mejora hasta que supera evaluaciones reproducibles y revisión adecuada.
- **Computación segura:** los nodos voluntarios reciben únicamente trabajos
  públicos y no sensibles, con límites configurables por su propietario.
- **Transparencia:** se documentan capacidades reales, riesgos, procedencia de
  datos y resultados; no se afirman entrenamientos ni mejoras no realizados.
- **Acceso responsable al conocimiento:** se respetan licencias, permisos,
  privacidad y controles de acceso de las fuentes.

## Documentación

- [Arquitectura](docs/ARCHITECTURE.md)
- [Hoja de ruta](docs/ROADMAP.md)
- [Estado de implementación](docs/PROGRESS.md)
- [Registro de decisiones](docs/DECISIONS.md)
- [Política de datos](docs/DATA_POLICY.md)
- [Modelo de seguridad](docs/SECURITY.md)
- [Gobernanza](docs/GOVERNANCE.md)
- [Licencia](LICENSE)

## Desarrollo local

Requiere Python 3.11 o posterior y Docker Compose para PostgreSQL:

```powershell
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
docker compose up -d db
python -m alembic upgrade head
python -m uvicorn ai_home.api:app --app-dir services/api/src --reload
```

La API local queda disponible en `http://127.0.0.1:8000`; la documentación
interactiva está en `/docs`. El endpoint `/health` es una comprobación de
proceso y no indica que PostgreSQL u otros servicios estén listos.

La configuración `AI_HOME_DATABASE_URL` admite PostgreSQL en el entorno
anterior y SQLite para pruebas/localización rápida. La aplicación no crea ni
modifica tablas automáticamente al arrancar: aplicar migraciones explícitas.
En producción se rechaza SQLite.

### Identidad inicial

- `POST /api/v1/auth/register` crea una cuenta con contraseña de al menos 12
  caracteres.
- `POST /api/v1/auth/login` devuelve una sesión bearer opaca; solo se guarda el
  hash SHA-256 del token y la contraseña se almacena con Argon2id.
- `GET /api/v1/users/me` devuelve exclusivamente la cuenta asociada al token.
- `POST /api/v1/auth/logout` revoca la sesión.

Todavía no hay recuperación de contraseña, verificación de correo, límite de
intentos, administración de cuentas ni interfaz web. No expongas esta API a
Internet: las protecciones operativas de producción aún no están completas.

### Pruebas

python -m pytest
```

Las pruebas de dominio/API usan SQLite en memoria; CI cubre Python 3.11 y 3.12.
Esto todavía no sustituye pruebas de concurrencia ni integración específica
con PostgreSQL.

## Estado y límites

Consulta [PROGRESS.md](docs/PROGRESS.md). Los datos personales no se distribuyen
a nodos voluntarios; el scheduler y los nodos todavía no están implementados.
El repositorio de GitHub sigue siendo privado. Apache-2.0 se aplica al código
de este repositorio; modelos, pesos y datasets tienen licencias propias que
deben comprobarse individualmente.
