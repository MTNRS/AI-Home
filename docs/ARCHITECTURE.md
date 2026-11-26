# Arquitectura inicial

## Objetivo y límites

AI@Home busca crecer hacia una plataforma de IA colaborativa, multimodal y
distribuida. La arquitectura descrita aquí es una dirección incremental, no la
afirmación de que esos subsistemas ya existan.

La prioridad es construir primero un núcleo pequeño y auditable: identidad,
autorización, trabajos de cómputo no sensibles y contabilidad de créditos. Los
límites de seguridad y datos se diseñan antes de agregar aprendizaje
colaborativo, rastreo web o autoevolución.

## Topología inicial

```text
Aplicación de escritorio experimental
    |                       |
    |                       +-- BYOK directo --> Proveedor externo
    v
API FastAPI <--- opt-in --- Candidaturas privadas pendientes
              |
  +-----------+------------------+
  |                              |
Módulos de dominio          PostgreSQL
  |                         fuente de verdad
  +-- identidad                 |
  +-- trabajos                  +-- usuarios y permisos
  +-- créditos                  +-- estado de trabajos
  +-- evaluación                +-- libro mayor inmutable
  +-- conocimiento               +-- auditoría
              |
      Coordinación de tareas
   (Redis opcional; sin autoridad
       sobre saldos o permisos)
              |
    Nodos voluntarios aislados
```

### Monolito modular primero

El primer backend será una aplicación desplegable, dividida en módulos con
interfaces explícitas. No se separarán servicios solo por anticipar escala:
esa separación añade fallos de red y complejidad operativa. Los módulos podrán
extraerse a servicios independientes cuando haya necesidades medidas y límites
de confianza que lo justifiquen.

## Límites de dominio

- **Identidad y autorización:** cuentas, sesiones, credenciales y permisos por
  recurso. Supabase Auth gestiona las credenciales; FastAPI verifica sus JWT
  con JWKS y mantiene una proyección mínima para las relaciones de datos. Las
  comprobaciones de autorización se realizan en el servidor en cada operación.
- **Trabajos y scheduler:** ciclo de vida de trabajos, asignación, expiración,
  reintentos y verificación de resultados.
- **Créditos:** libro mayor transaccional, reservas, liquidaciones y reembolsos.
  Los saldos se derivan de asientos confirmados, no de mensajes de una cola.
- **Inferencia y modelos:** el escritorio ofrece BYOK directo para OpenAI,
  Anthropic, Gemini y endpoints OpenAI-compatibles; un proxy autenticado usa
  el proveedor configurado por quien opera la instancia. AI@Home aún no ejecuta
  ni distribuye modelos propios; su origen, versión y licencia deberán
  registrarse al integrar un runtime.
- **Conocimiento:** conectores autorizados, procedencia, indexación y
  recuperación. Los índices globales no incluyen datos privados.
- **Privacidad y memoria:** datos personales separados por usuario; su acceso
  y eliminación se controlan independientemente del conocimiento global.
- **Evaluación y evolución:** experimentos reproducibles y aislados. Una
  propuesta generada automáticamente no puede cambiar permisos, privacidad,
  contabilidad o autenticación sin revisión y autorización explícitas.

Estos límites guían la implementación. Identidad Supabase opcional, identidad
local de desarrollo, trabajos, créditos, BYOK y el proxy de inferencia tienen
componentes iniciales; el despliegue de FastAPI, un runtime propio,
conocimiento, memoria y evolución siguen siendo áreas de diseño o pendientes.

## Fronteras de datos y confianza

Clasificación inicial:

1. **Público:** código, documentación y conjuntos de datos autorizados para el
   propósito concreto.
2. **Operativo:** identificadores de trabajos, métricas y telemetría mínima.
3. **Personal privado:** conversaciones, archivos y memoria de una cuenta;
   las conversaciones candidatas se guardan solo por opt-in explícito y quedan
   pendientes de revisión.
4. **Secreto:** credenciales, claves y tokens.

Los nodos voluntarios son entornos no confiables. Solo podrán recibir tareas
que no requieran datos personales, secretos ni acceso a redes o recursos no
autorizados. Las tareas sensibles se ejecutan localmente o en infraestructura
explícitamente confiable. No se asumirá que cifrado o anonimización eliminan
los riesgos de inferencia, memorización o extracción.

Todo acceso a recursos privados debe tener un principal autenticado, una
comprobación de pertenencia/autorización y una prueba automatizada que rechace
acceso entre usuarios. Los registros de auditoría no deben copiar contenido
privado.

Las claves BYOK se mantienen en el almacén seguro local y nunca se envían a
AI@Home. El chat ordinario no se persiste; una candidatura solo llega al
servidor después de consentimiento por conversación y una acción explícita.
Las candidaturas actuales se guardan como JSON sin cifrado de campo, pueden
ser leídas por el operador de la instancia y no se usan automáticamente para
entrenamiento. Un endpoint administrativo con secreto configurado permite
revisión humana y snapshots versionados con hash; no hay interfaz administrativa
ni entrenamiento que consuma esos datasets.

## Ciclo de trabajo distribuido

El flujo previsto es:

1. Un usuario o proceso autorizado crea un trabajo tipado y limitado.
2. El scheduler verifica elegibilidad, recursos y política de datos.
3. Se asigna una concesión temporal con identificador único y vencimiento.
4. El nodo ejecuta el trabajo dentro de restricciones locales configuradas
   por su propietario.
5. El servidor valida el resultado, idempotencia y límites de recursos antes
   de aceptarlo.
6. La liquidación de créditos se hace transaccionalmente solo para trabajo
   aceptado; resultados repetidos no generan doble crédito.
7. El usuario puede consultar el estado y la auditoría permitida del trabajo.

El único handler disponible actualmente es `math.sum.v1`; otros tipos de
trabajo no se presentan como funcionales. La ejecución en nodos voluntarios no
debe permitir acceso arbitrario al host.
El diseño de aislamiento del proceso, verificación del binario y actualización
del cliente es un requisito previo al despliegue de trabajo no confiable.

## Libro mayor de créditos

- Mantener asientos append-only en PostgreSQL con claves foráneas y
  restricciones de unicidad.
- Exponer operaciones de escritura solo desde servicios autorizados; los
  modelos ORM no sustituyen controles de acceso a la base de datos.
- Hacer reserva, liquidación y reembolso en transacciones.
- Asociar cada operación a una clave idempotente y a su trabajo o solicitud.
- Derivar balances de asientos confirmados o mantener proyecciones
  reconstruibles.
- No crear un mercado monetario, compra de créditos ni criptomoneda.
- No acreditar por tiempo conectado; la política de recompensa necesita
  resultados validados y resistencia al fraude.

Las reglas de equivalencia entre tipos de hardware deberán calibrarse y
publicarse antes de activar recompensas.

## Conocimiento y aprendizaje

Se separan explícitamente:

- **Paramétrico:** lo incorporado a los parámetros del modelo durante el
  entrenamiento.
- **Recuperable:** contenido con procedencia guardado en índices autorizados.
- **En tiempo real:** información consultada en fuentes permitidas para una
  solicitud.
- **Memoria personal:** información privada accesible solo a su propietario y
  nunca promovida automáticamente a datasets globales.

La contribución a conocimiento o entrenamiento requiere consentimiento
específico, propósito definido, registro de procedencia y un mecanismo de
retirada aplicable. La autorización para una conversación no implica
autorización para entrenamiento.

## Evolución y despliegues

El Evolution Engine se desarrollará como pipeline experimental: propuesta,
entorno aislado, pruebas reproducibles, evaluación independiente y decisión
humana para cambios de alto impacto. No tendrá credenciales de producción ni
capacidad de modificar sus propios controles de seguridad.

Cada versión candidata debe identificar código, modelo base, datasets
autorizados, configuración, semilla, métricas, recursos y resultados. Se
conserva una versión estable anterior para reversión. No se publican resultados
preliminares como mejoras demostradas.

## Tecnologías previstas

- Python y FastAPI para la API inicial.
- PostgreSQL como almacenamiento transaccional y fuente de verdad.
- SQLAlchemy y Alembic para persistencia y migraciones.
- Redis opcional para coordinación no autoritativa.
- React, TypeScript y Vite para una interfaz web posterior.
- PyTorch y herramientas compatibles con los modelos escogidos para
  entrenamiento/evaluación posterior.
- Docker Compose para desarrollo local cuando los primeros servicios estén
  implementados.

Las versiones y dependencias se fijarán al crear cada componente, con pruebas
de compatibilidad y licencias.

## Criterios para extraer un servicio

Separar un módulo del backend solo cuando exista una razón verificable, como
aislamiento de fallos, perfil de escalado distinto, frontera de confianza o
necesidad de desplegarlo de forma independiente. La extracción debe preservar
autorización, trazabilidad, idempotencia y pruebas de privacidad.

## Foundation Model — implementación inicial (2026-11-26)

Se añade un subsistema propio en `ai_home.foundation` con BPE desde cero, Transformer RoPE/GQA, entrenamiento, checkpoints e inferencia. El modelo byte-level y las funcionalidades anteriores se conservan. La integración de escritorio añade un proveedor independiente para checkpoints propios.

**Estado: parcial y bloqueado para validación neuronal.** Cuatro pruebas sin PyTorch pasan; no se ha completado entrenamiento, generado un checkpoint ni ejecutado inferencia nueva. PyTorch/pytest no están disponibles y su instalación falla. La escritura de ramas en GitHub devuelve 403 y Git local no puede crear `.git/index.lock`; no se han creado commits.

Consultar [auditoría y matriz completa](FOUNDATION_AUDIT.md), [modelo](FOUNDATION_MODEL.md), [entrenamiento](TRAINING.md) y [pruebas](TESTING.md). Aprendizaje continuo, distribución, créditos de entrenamiento, RAG público, evolución y paneles completos siguen pendientes; no se reduce el objetivo ni se declaran demostrados.



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

## Motores de evaluación y evolución

`foundation.evaluation` carga un checkpoint propio y ejecuta probes declarativos sin ejecutar código generado ni llamar a jueces externos. `foundation.evolution` conserva una propuesta previa, crea dos candidatos mediante continual y compara evidencia del mismo protocolo. Solo emite propuesta; no tiene rutas de escritura al modelo activo, permisos, créditos o infraestructura.

## Trabajo Foundation tipado

foundation.tasks define el contrato acotado de tokenización pública. API, jobs y NodeAgent reutilizan autenticación, idempotencia y leases. El servidor reejecuta para verificar; no se añade un intérprete remoto ni se distribuyen exportaciones restringidas. La asignación exige capacidad declarada y un mínimo conservador de RAM.

## Entrenamiento por shards experimental

foundation.distributed define tensores FP32 acotados y gradientes causales por shard. API/jobs asignan solo a nodos compatibles con opt-in; el servidor recalcula antes de aceptar. El coordinador aplica promedio ponderado por tokens, clipping y AdamW. Los checkpoints candidatos preservan trazabilidad y artefactos de ronda. El protocolo de referencia es CPU, no una solución eficiente para GPUs heterogéneas o redes domésticas.
