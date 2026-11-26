# AI@Home

Estado experimental Foundation: [inventario de implementación, pruebas y pendientes](docs/FOUNDATION_STATUS.md). [PR de desarrollo #1](https://github.com/MTNRS/AI-Home/pull/1), todavía en borrador. Hay entrenamiento textual y visual propio desde cero, sin calidad conversacional ni comprensión visual general demostradas.

La [vía visual experimental](docs/VISION_PREFIX.md) permite entrenar pares PNG/JPEG–texto autorizados, reanudar checkpoints FP32/FP16/BF16 y ejecutar inferencia local con streaming desde el escritorio. TRAINING incluye entrenamiento visual con métricas y cancelación; FOUNDATION MODEL incluye prueba de imágenes. El único aprendizaje visual medido por ahora es un experimento de tres colores sólidos: 6/6 intensidades reservadas, control blanco 2/6; no representa fotografías ni objetos reales. Consulta la documentación para formatos, comandos reproducibles y límites.

AI@Home es un proyecto comunitario de inteligencia artificial abierta, con el
objetivo de permitir investigación colaborativa y computación voluntaria sin
comprometer la privacidad de sus participantes.

El proyecto está en desarrollo temprano. Ya incluye una API inicial de
identidad y sesiones, un protocolo experimental de nodos voluntarios y un
núcleo interno de libro mayor, además de un primer cliente de escritorio y el
código de un LLM experimental entrenable desde inicialización aleatoria.
Todavía no hay un checkpoint entrenado ni se incluye un modelo ya preparado;
los trabajos de nodos no están conectados a créditos.

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

### Supabase Auth

El backend Python usa Supabase Auth para registro, inicio/cierre de sesión y
verificación de tokens. `@supabase/server` es para servidores JavaScript y no
se instala en este backend; la integración Python usa HTTPX y verifica los JWT
con las claves públicas JWKS del proyecto. Configura `SUPABASE_URL` y
`SUPABASE_PUBLISHABLE_KEY` en el entorno del backend. Configura también
`SUPABASE_SECRET_KEY` solo en el backend para que la eliminación de cuenta
pueda borrar la identidad de Supabase; nunca la incluyas en el escritorio,
repositorio o mensajes. `SUPABASE_JWKS_URL` se deriva de la URL si se omite.
En desarrollo, si no configuras Supabase, el backend conserva el proveedor
local Argon2 para pruebas; producción exige Supabase y su clave secreta.

Los perfiles mínimos de AI@Home conservan el UUID de Supabase para relacionar
nodos y aportes. Para que esos datos sobrevivan al reinicio y estén compartidos,
configura `AI_HOME_DATABASE_URL` con la conexión PostgreSQL administrada de
Supabase y aplica las migraciones. Supabase aloja Auth y PostgreSQL, pero no
ejecuta este backend FastAPI: para que otras personas se conecten también hay
que desplegar la API en un servicio compatible y configurar su URL en el
cliente. El proyecto aún no está desplegado.

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

Todavía no hay recuperación de contraseña, límite de intentos, administración
de cuentas ni interfaz web. La verificación de correo se configura en Supabase.
El
protocolo de nodos solo permite `math.sum.v1` y no acredita créditos. El
cliente de escritorio es experimental y no se distribuye todavía como
instalador. No expongas esta API a Internet: las protecciones operativas de
producción aún no están completas.

### Cliente de escritorio experimental

Con Python 3.11+ y Tk instalado (incluido normalmente en Windows), instala el
proyecto y ejecuta:

```powershell
python -m pip install -e ".[llm]"
python -m ai_home.desktop
```

La pantalla **Chat** mantiene centrada arriba una esfera sombreada que se
orienta suavemente hacia el puntero y se queda quieta cuando este no se mueve.
Muestra las respuestas de la IA a la izquierda y tus mensajes a la derecha,
sin fondos de color en los mensajes. La barra superior da acceso a **Conectar
IA**, **Cuenta** y **Colaborar** sin mezclar esas opciones con el chat. En
**Conectar IA** puedes
usar el modelo local experimental (opción predeterminada), elegir OpenAI,
Anthropic, Google Gemini o configurar manualmente un endpoint
OpenAI-compatible. Las claves BYOK se guardan únicamente en el almacén seguro
del sistema operativo, se pueden eliminar desde la aplicación y nunca se
guardan en las preferencias, se envían al servidor AI@Home ni se usan para
entrenamiento. Si el almacén seguro no está disponible, no hay fallback a
texto plano. Las solicitudes directas salen del escritorio al proveedor
elegido; el proveedor externo procesa el contenido según sus propias
condiciones. El modelo local no necesita clave ni envía el chat a un servicio
externo. Si todavía no hay checkpoint, la aplicación explica cómo prepararlo;
no cambia silenciosamente a un proveedor externo.

No hace falta una cuenta para abrir una sesión de invitado y conversar con un
proveedor configurado o usar la memoria personal local. En **Cuenta**, elige
**Continuar como invitado**. Las funciones de la comunidad, como usar el
proveedor del servidor, colaborar con nodos o enviar candidaturas de
entrenamiento, siguen requiriendo iniciar sesión.

La memoria personal se guarda localmente como notas Markdown compatibles con
Obsidian, en la carpeta de configuración de AI@Home del usuario. **Memoria
local** permite añadir recuerdos y enlazarlos con otros; los nodos y enlaces se
visualizan en el fondo del chat. En cada pregunta se buscan coincidencias en el
equipo y solo las notas relevantes se añaden al contexto enviado al proveedor
seleccionado. Esto aporta contexto a la inferencia, no entrena los pesos del
modelo. Los archivos de memoria no se sincronizan con el servidor.

También se puede seleccionar el proveedor configurado por una instancia
AI@Home autenticada. Para ello, la persona operadora configura
`AI_HOME_CHAT_PROVIDER_URL` y `AI_HOME_CHAT_PROVIDER_MODEL`; la clave opcional
`AI_HOME_CHAT_PROVIDER_API_KEY` permanece en el servidor. El proxy envía los
mensajes al proveedor de esa instancia y no los añade a historial persistente.

La aplicación no guarda normalmente el historial en disco ni en el servidor.
Por separado, una persona puede activar dos confirmaciones para enviar una
conversación completa al servidor AI@Home como **candidata** a entrenamiento.
La interfaz advierte que el servidor almacenará los mensajes en su base de
datos configurada para revisión, que su operador podrá acceder y que no se
garantiza retirar su influencia de pesos publicados. La selección está apagada
por defecto; se requieren cuenta, revisión del contenido y confirmación de que
los términos del proveedor permiten reutilizarlo. Cada envío se crea como
`pending_review` y necesita revisión humana para ser elegible; no se entrena
automáticamente. Las candidaturas propias se pueden consultar o retirar
mediante la aplicación o la API:

- `POST /api/v1/training/conversations`: crear una candidatura con
  consentimiento/versionado y procedencia proveedor/modelo.
- `GET /api/v1/training/conversations`: consultar metadatos de candidaturas
  propias paginadas con `limit` y `offset`.
- `GET /api/v1/training/conversations/{contribution_id}`: leer el contenido de
  una candidatura propia.
- `DELETE /api/v1/training/conversations/{contribution_id}`: borrar una
  candidatura antes de congelarse en un dataset.
- `GET /api/v1/admin/training/conversations?status=pending_review|eligible`:
  revisión de candidaturas (credencial administrativa requerida).
- `POST /api/v1/admin/training/conversations/{contribution_id}/review`: decisión
  humana; para aprobar exige confirmar términos, revisión de datos sensibles y
  derechos de reutilización.
- `POST /api/v1/admin/training/datasets`: congelar hasta 100 candidaturas
  elegibles en una versión inmutable con hash SHA-256 del manifiesto.
- `GET /api/v1/admin/training/datasets` y
  `GET /api/v1/admin/training/datasets/{dataset_id}`: consultar manifiestos y
  registros congelados (credencial administrativa requerida).

La API bloquea patrones comunes de credenciales en mensajes y endpoints, pero
no garantiza detectar datos personales ni todos los secretos; la revisión
humana sigue siendo necesaria. La revisión y congelación están disponibles solo
mediante una API administrativa protegida por `AI_HOME_TRAINING_REVIEW_TOKEN`
(secreto aleatorio de al menos 32 caracteres) y no tienen aún una interfaz
administrativa. Los datasets congelados son snapshots inmutables con hash y no
se eliminan al borrar la cuenta; el pipeline local experimental aún no consume
esos snapshots. La pantalla **Colaborar** es independiente y requiere
iniciar sesión y aceptar expresamente antes de registrar/iniciar
un nodo. Las credenciales de nodo también usan el almacén seguro del sistema
operativo; las preferencias no secretas quedan en el directorio de
configuración del usuario.

### Entrenamiento experimental del LLM propio

El modelo es un pequeño GPT byte-level implementado con PyTorch, inicializado
sin pesos preentrenados. El importador de Wikipedia crea ejemplos de pregunta
y respuesta basados en las introducciones de artículos; el importador OASST1
selecciona conversaciones revisadas en español bajo licencia Apache-2.0. Ambos
producen manifiestos y registros de atribución. No se utilizan conversaciones
de invitados ni chats locales como datos de entrenamiento.

Descarga manualmente los archivos oficiales: un dump de artículos de Wikipedia
en español desde [Wikimedia Dumps](https://dumps.wikimedia.org/eswiki/) y el
archivo
[`2023-04-12_oasst_ready.messages.jsonl.gz`](https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.messages.jsonl.gz)
de la exportación
[OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1).
Después prepara ambos corpus:

```powershell
python -m ai_home.wikipedia_corpus `
  --dump C:\datasets\eswiki-latest-pages-articles-multistream.xml.bz2 `
  --output C:\datasets\wiki-es.txt `
  --source-url https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles-multistream.xml.bz2

python -m ai_home.conversation_corpus `
  --source C:\datasets\2023-04-12_oasst_ready.messages.jsonl.gz `
  --output C:\datasets\oasst-es.txt

python -m ai_home.train_llm `
  --corpus C:\datasets\wiki-es.txt C:\datasets\oasst-es.txt `
  --output "$env:APPDATA\ai-home\models\aihome-small.pt" `
  --steps 10000 `
  --device auto
```

El entrenamiento muestrea ambos corpus por igual para que el dump enciclopédico
no eclipse el diálogo. Se puede omitir `--output` para guardar el checkpoint
en la carpeta de configuración del usuario que consulta automáticamente el
escritorio. Antes de distribuir corpus o pesos derivados, revisa las licencias,
las atribuciones y posibles excepciones por artículo. Un modelo pequeño y unos
pocos pasos no garantizan una conversación coherente: esta implementación es
un primer experimento local, no un asistente competitivo; hay que evaluar sus
respuestas y entrenarlo con datos conversacionales autorizados suficientes.

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

## Foundation Model — implementación inicial (2026-11-26)

**Registro histórico del primer hito.** Los resultados y pendientes de este apartado corresponden a aquella ejecución; para el estado actual usa [FOUNDATION_STATUS.md](docs/FOUNDATION_STATUS.md) y [PROGRESS.md](docs/PROGRESS.md).

Se añade un subsistema propio en `ai_home.foundation` con BPE desde cero, Transformer RoPE/GQA, entrenamiento, checkpoints e inferencia. El modelo byte-level y las funcionalidades anteriores se conservan. La integración de escritorio añade un proveedor independiente para checkpoints propios.

**Estado: parcial, mecánica neuronal verificada en CPU (2026-11-26).** La suite ejecutada obtiene 127 pruebas aprobadas y 5 omitidas (CUDA y PostgreSQL). El smoke entrena 27.808 parámetros desde cero durante ocho pasos, reanuda desde el paso cuatro, guarda/carga checkpoints y genera texto real todavía incoherente. Cambio L2 de embeddings tras reanudación: 0,02502243; pérdida de validación: 5,615235. PyTorch disponible: 2.15.0+cpu. La integración visual y distribución siguen sin demostrar; CUDA se verificó posteriormente en un entorno separado (ver CUDA_VALIDATION.md). Las restricciones de instalación descritas anteriormente ya no bloquean CPU; la publicación remota sigue pendiente.

Consultar [auditoría y matriz completa](docs/FOUNDATION_AUDIT.md), [modelo](docs/FOUNDATION_MODEL.md), [entrenamiento](docs/TRAINING.md) y [pruebas](docs/TESTING.md). Aprendizaje continuo, distribución, créditos de entrenamiento, RAG público, evolución y paneles completos siguen pendientes; no se reduce el objetivo ni se declaran demostrados.



## Chat HTTP con Foundation (2026-11-26)

Configurar `AI_HOME_CHAT_BACKEND=foundation` (valor predeterminado),
`AI_HOME_FOUNDATION_CHECKPOINT` con la ruta al checkpoint propio y
`AI_HOME_FOUNDATION_DEVICE=cpu`, `cuda` o `auto`. El límite de salida se configura
con `AI_HOME_FOUNDATION_MAX_NEW_TOKENS` (1–4096); el presupuesto de carga con
`AI_HOME_FOUNDATION_MEMORY_BUDGET_BYTES`. El checkpoint lo elige el operador,
no una ruta arbitraria enviada por el cliente.

La ruta autenticada `/api/v1/chat/completions` usa inferencia neuronal local.
Un checkpoint ausente, corrupto o una dependencia no disponible devuelve 503;
no hay fallback externo. Cada petición crea su propio modelo/cache y no guarda
mensajes ni los incorpora al entrenamiento. Una generación simultánea por
proceso; se devuelve 503 y Retry-After cuando está ocupado. Desplegar un único
worker por dispositivo para respetar este límite. La carga por petición es
conservadora y puede añadir latencia; caché de pesos compartida pendiente.

Para conservar el proxy externo, configurar explícitamente
`AI_HOME_CHAT_BACKEND=openai-compatible` además de sus variables anteriores.
Esta selección explícita es necesaria al actualizar instalaciones del proxy.
El endpoint HTTP devuelve una respuesta completa; streaming/cancelación HTTP
siguen pendientes (el generador neuronal ya soporta ambos localmente).

## Validación GPU completada — 2026-11-26

Entrenamiento, reanudación e inferencia propios verificados en RTX 4070 SUPER con FP32/FP16/BF16. Suite del entorno CUDA: 139 aprobadas, 4 omitidas (PostgreSQL). Checkpoint y métricas reales conservados. No se atribuye calidad conversacional. Comandos y límites: [docs/CUDA_VALIDATION.md](docs/CUDA_VALIDATION.md).

## Entrenamiento experimental en nodos

Existe una demostración reproducible de gradientes reales aportados por dos identidades de nodo mediante procesos separados y HTTP loopback. Agrega gradientes, actualiza pesos, guarda checkpoints y genera texto. Solo se ha verificado en un equipo y CPU; no se reclama rendimiento distribuido multihost. Véase [DISTRIBUTED_TRAINING.md](docs/DISTRIBUTED_TRAINING.md).

La API experimental tambien admite `foundation.evaluate.v1` para perdida causal
sobre pesos y datos publicos acotados, con recomputacion CPU antes de conceder
creditos. Requiere habilitacion independiente en servidor y trabajador; no esta
activada por defecto ni disponible aun como selector del escritorio. La prueba
actual de esta tarea usa HTTP en proceso, no sockets ni equipos distintos.


## Créditos por contribuciones verificadas

Los trabajos Foundation pueden financiar recompensas mediante reservas del solicitante. La verificación del resultado y el abono al contribuyente comparten transacción; cancelaciones y fallos terminales devuelven fondos. Consultas de saldo/movimientos propios y migración 0006 disponibles. Véase [CREDITS.md](docs/CREDITS.md). El cobro de inferencia sigue pendiente.


### Public knowledge retrieval increment (2026-11-26)

Added a versioned reviewed-public TF-IDF index with provenance, bounded context
packing, and opt-in retrieval in authenticated Foundation chat. Retrieved text
and personal memories use user-level context, not system instructions. This is
retrieval, not weight training or a demonstrated improvement in model quality.
See [Knowledge retrieval](docs/KNOWLEDGE.md) for commands,
permissions, tests and limitations. Network connectors and desktop references
remain outstanding.


### Model lifecycle registry (2026-11-26)

Added local content-addressed candidate storage with real retained-checkpoint
evaluation, explicit approval gates, transactional activation history and rollback.
Foundation API serving can select the active approved registry version. Direct
checkpoint mode remains available and is not an approval claim. See
[Model registry](docs/MODEL_REGISTRY.md) for commands, reproducibility evidence and limitations.
