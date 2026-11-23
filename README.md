# AI@Home

AI@Home es un proyecto comunitario de inteligencia artificial abierta, con el
objetivo de permitir investigación colaborativa y computación voluntaria sin
comprometer la privacidad de sus participantes.

El proyecto está en desarrollo temprano. Ya incluye una API inicial de
identidad y sesiones, un protocolo experimental de nodos voluntarios y un
núcleo interno de libro mayor, además de un primer cliente de escritorio. AI@Home
aún no aloja sus propios modelos de inferencia/chat y los trabajos de nodos no
están conectados a créditos.

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
- [Protocolo inicial de nodos](docs/NODE_PROTOCOL.md)
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
- `GET /api/v1/users/me/export` exporta los campos básicos de la cuenta.
- `DELETE /api/v1/users/me` elimina la cuenta y sus sesiones/nodos/trabajos
  después de confirmar la contraseña.

Los nodos pueden registrarse con `POST /api/v1/nodes`, y los trabajos
deterministas `math.sum.v1` con `POST /api/v1/jobs` y consentimiento explícito
para compartir sus datos con nodos. El protocolo y sus limitaciones están en
[NODE_PROTOCOL.md](docs/NODE_PROTOCOL.md). Este primer ejecutor no acredita
créditos y no ejecuta código de usuario.

Todavía no hay recuperación de contraseña, verificación de correo, límite de
intentos, administración de cuentas ni interfaz web. El
protocolo de nodos solo permite `math.sum.v1` y no acredita créditos. El
cliente de escritorio es experimental y no se distribuye todavía como
instalador. No expongas esta API a Internet: las protecciones operativas de
producción aún no están completas.

### Cliente de escritorio experimental

Con Python 3.11+ y Tk instalado (incluido normalmente en Windows), instala el
proyecto y ejecuta:

```powershell
python -m pip install -e .
python -m ai_home.desktop
```

La pestaña **Usar IA** conecta por defecto directamente a un proveedor
compatible con la API OpenAI en `http://localhost:11434/v1` (por ejemplo,
Ollama instalado y ejecutándose localmente). También permite usar el proveedor
configurado por la instancia AI@Home: inicia sesión y selecciona **Usar el
proveedor configurado en el servidor AI@Home**. Para habilitar ese modo, la
persona que opera el servidor debe configurar `AI_HOME_CHAT_PROVIDER_URL` y
`AI_HOME_CHAT_PROVIDER_MODEL`; la clave opcional se configura en
`AI_HOME_CHAT_PROVIDER_API_KEY` y no se envía al cliente.

Los mensajes no se guardan en el historial local ni en la base de datos de
AI@Home; en ambos modos se envían al proveedor elegido/configurado. Usa solo
un proveedor en el que confíes y no envíes secretos ni datos personales sin
autorización. La pestaña **Colaborar** es independiente y requiere iniciar
sesión y aceptar expresamente antes de registrar/iniciar un nodo. Las
credenciales de nodo se guardan con el almacén seguro del sistema operativo;
preferencias no secretas se guardan en el directorio de configuración del
usuario.

### Pruebas

```powershell
python -m pytest
```

Las pruebas locales de dominio/API usan SQLite en memoria. GitHub Actions está
configurada para probar Python 3.11 y 3.12 e integración/concurrencia con
PostgreSQL 17; las pruebas automatizadas del cliente de escritorio y el proxy
de chat pasaron en ambas versiones.

## Estado y límites

Consulta [PROGRESS.md](docs/PROGRESS.md). Solo los datos de trabajos creados
con consentimiento explícito pueden enviarse a nodos; no deben incluir datos
personales, secretos ni valores confidenciales. El repositorio de GitHub sigue
siendo privado. Apache-2.0 se aplica al código de este repositorio; modelos,
pesos y datasets tienen licencias propias que deben comprobarse individualmente.
