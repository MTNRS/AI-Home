# Estado de implementación

Última actualización: 2026-11-23

## Fase actual

**Fases 3, 5 y 8 — Nodo voluntario, cliente BYOK y captura consentida
experimental; producción pendiente.**

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
- Integración opcional con Supabase Auth: registro/login mediante Auth REST,
  verificación de JWT con JWKS del proyecto, revocación local de sesiones y
  proyección local del UUID/correo sin guardar contraseñas. La clave secreta se
  limita a eliminación administrativa de cuentas.
- Migración `0005` permite perfiles con credenciales gestionadas externamente;
  el modo local de desarrollo conserva Argon2id y los tokens se almacenan como
  hashes.
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
- Primera aplicación de escritorio Tk con cuenta, chat y aportación de nodos
  claramente opcional; tokens de nodo en el almacén seguro del sistema
  operativo.
- Adaptadores directos de escritorio para OpenAI, Anthropic, Google Gemini y
  endpoints OpenAI-compatibles. Claves guardadas en keyring por proveedor y
  endpoint; no se incluyen en preferencias ni se envían a AI@Home. Endpoints
  oficiales fijos y endpoint compatible remoto sujeto a HTTPS.
- UI BYOK advierte del procesamiento por terceros y permite guardar/eliminar
  claves con manejo explícito de errores del almacén seguro, sin fallback a
  texto plano.
- Migración `0003` y API autenticada para candidaturas de conversaciones:
  consentimiento/versionado/procedencia, opt-in doble apagado por defecto,
  filtro de secretos comunes, aislamiento por cuenta, consulta propia,
  retirada física antes de congelación y eliminación al borrar la cuenta.
- Migración `0004` y API de revisión protegida por secreto administrativo:
  aprobación exige checklist de términos/datos sensibles/derechos; rechazos
  eliminan el contenido. Candidaturas elegibles se pueden congelar en datasets
  versionados de hasta 100 muestras con snapshot y SHA-256 reproducible.
  Retiradas se sincronizan con el lock de fila; el snapshot congelado persiste
  al borrar la cuenta y no es reversible.
- La interfaz permite enviar una conversación solo tras opt-in y revisión
  expresa; permite consultar o retirar candidaturas propias. La revisión de
  contenidos se opera solo mediante API administrativa; no hay UI de revisión.
  Los datasets congelados no se consumen aún en entrenamiento.
- Proxy de chat opcional y autenticado en API para un proveedor
  OpenAI-compatible configurado por la instancia; clave solo del lado servidor.
- Inventario local verificado: 16 hilos lógicos, 31 GiB de RAM y una NVIDIA
  GeForce RTX 4070 SUPER con 12 GiB de VRAM; la detección usa `nvidia-smi`.
- CI aprobada previamente en Python 3.11/3.12 con PostgreSQL 17 para la línea
  base; los cambios BYOK/candidaturas aún requieren ejecución en CI/PostgreSQL.
- Suite local más reciente antes de este cambio: `python -m pytest -q` — 103
  pruebas aprobadas y 4 pruebas PostgreSQL omitidas por no haber servidor
  disponible localmente.
- Suite local tras implementar el pipeline del LLM: `python -m pytest -q` —
  120 aprobadas y 4 omitidas por no haber servidor PostgreSQL local.
- PyTorch no estaba instalado ni había corpus local al iniciar este cambio. Se
  añadió como extra opcional; la validación usó una versión CPU, no una sesión
  de entrenamiento real en GPU. La RTX 4070 SUPER de 12 GiB permite investigar
  un modelo pequeño, no asumir que se puede entrenar aquí un LLM competitivo.
- Implementado un GPT byte-level pequeño con pesos inicializados aleatoriamente,
  entrenamiento local con PyTorch opcional y carga de checkpoints locales. La
  app lo selecciona por defecto y muestra un error explícito hasta que exista
  un checkpoint; no hace fallback silencioso a proveedores externos.
- Importadores experimentales para Wikipedia en español y para conversaciones
  revisadas en español de OASST1 (Apache-2.0). Cada corpus lleva manifiesto,
  hash al entrenar y registros de atribución; el entrenamiento puede mezclar
  corpus con muestreo equilibrado. No se descarga contenido automáticamente.
- El código de entrenamiento y los importadores están implementados. Se
  verificó la importación real de OASST1 (3.054 ejemplos conversacionales en
  español tras filtros), pero no se entrenó/evaluó un checkpoint con ese corpus.
  El modelo byte-level es una prueba de concepto; no se afirma que pueda
  mantener conversaciones coherentes.
- Las pruebas incluyen adaptadores de proveedores con transportes simulados,
  keyring aislado, consentimiento, bloqueo de secretos comunes, aislamiento y
  retirada. No se hicieron llamadas de proveedor con credenciales reales ni
  revisión visual manual del escritorio.
- `python -m compileall` y `git diff --check` aprobados tras los cambios.

## Aún no implementado

- PostgreSQL no está disponible localmente; su integración y concurrencia se
  validan mediante GitHub Actions.
- Aún no están configurados el URL/clave de Supabase en un entorno de
  despliegue ni la contraseña de su PostgreSQL administrado. No se debe pegar
  la clave secreta en el chat; se configura únicamente como variable privada
  del backend.
- La API FastAPI no está desplegada. Supabase proporciona Auth y PostgreSQL,
  pero no aloja por sí solo este servidor Python; aún hace falta un servicio de
  hosting compatible para que los clientes externos usen la API comunitaria.
- Endurecimiento de producción de autenticación: límite de intentos y
  recuperación de cuenta. La verificación de correo se administra en Supabase.
- Detección de hardware en otros equipos y sistemas operativos; CUDA es
  opcional y no se presume.
- Verificación manual de la interfaz gráfica y publicación de instaladores
  Windows/Linux.
- Entrenamiento real con corpus autorizado, evaluación de calidad/seguridad y
  publicación del checkpoint. Los datasets congelados de conversaciones
  aportadas aún no alimentan el entrenamiento.
- Límites locales de CPU/RAM, controles de pausa/reanudación y sandbox
  reforzado para el nodo.
- Integración del libro mayor con trabajos verificados y autorización.
- Integración de modelos, chat o memoria.
- Rastreo web, RAG, uso de candidaturas en entrenamiento o motor de evolución.
- Interfaz web, instaladores de escritorio y operación de producción.
- TLS, rate limiting, CSRF si se añaden cookies, alertas y respuesta a
  incidentes antes de exposición pública.

Docker no está instalado o no está disponible en PATH en el entorno de
desarrollo actual; la verificación PostgreSQL se delega a GitHub Actions.

Las candidaturas se guardan en la base de datos configurada y esta versión no
cifra su contenido a nivel de campo. El filtro de secretos no es exhaustivo ni
es una detección de PII; no aportar contenido sensible. No hay un proceso que
entrene automáticamente. Las rutas administrativas de revisión y congelación
están protegidas por `AI_HOME_TRAINING_REVIEW_TOKEN`; el endpoint de revisión
expone contenido candidato a quien posea ese secreto. La API bloquea la
retirada de muestras ya congeladas y conserva snapshots aunque se borre la
cuenta de origen.

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

## Foundation Model — implementación inicial (2026-11-26)

Se añade un subsistema propio en `ai_home.foundation` con BPE desde cero, Transformer RoPE/GQA, entrenamiento, checkpoints e inferencia. El modelo byte-level y las funcionalidades anteriores se conservan. La integración de escritorio añade un proveedor independiente para checkpoints propios.

**Estado: parcial, mecánica neuronal verificada en CPU (2026-11-26).** La suite ejecutada obtiene 127 pruebas aprobadas y 5 omitidas (CUDA y PostgreSQL). El smoke entrena 27.808 parámetros desde cero durante ocho pasos, reanuda desde el paso cuatro, guarda/carga checkpoints y genera texto real todavía incoherente. Cambio L2 de embeddings tras reanudación: 0,02502243; pérdida de validación: 5,615235. PyTorch disponible: 2.15.0+cpu. La integración visual y distribución siguen sin demostrar; CUDA se verificó posteriormente en un entorno separado (ver CUDA_VALIDATION.md). Las restricciones de instalación descritas anteriormente ya no bloquean CPU; la publicación remota sigue pendiente.

Consultar [auditoría y matriz completa](FOUNDATION_AUDIT.md), [modelo](FOUNDATION_MODEL.md), [entrenamiento](TRAINING.md) y [pruebas](TESTING.md). Aprendizaje continuo, distribución, créditos de entrenamiento, RAG público, evolución y paneles completos siguen pendientes; no se reduce el objetivo ni se declaran demostrados.



## Integración HTTP Foundation — 2026-11-26

Implementada inferencia propia autenticada en el endpoint de chat, configuración de dispositivo/presupuesto/salida y exclusión mutua por proceso. Prueba real con entrenamiento y generación, rechazo de checkpoint corrupto/ausente y ausencia de fallback externo: 11 pruebas de chat aprobadas. Streaming HTTP, selección de versiones por usuario e interfaz de control completa siguen pendientes.

## Contribuciones hacia Foundation — 2026-11-26

Conectada exportación administrativa congelada a entrenamiento SFT local real. Verificados consentimiento, hash y revisión; el resultado es candidato de uso restringido, no publicación. Prueba extremo a extremo: crear contribuciones, aprobar, congelar, exportar, entrenar, guardar y recargar. Replay, fine-tuning incremental y promoción siguen pendientes.

## Aprendizaje incremental — 2026-11-26

Implementados candidatos desde checkpoints propios, BPE congelado, replay por lote, holdout fijo explícito, trazabilidad ancestral, bloqueo de contaminación exacta y propagación de restricciones. Experimento CPU real con informe en CONTINUAL_SMOKE_CPU.json. Promoción/rollback centralizado, replay restringido, capacidades lingüísticas y control general del olvido siguen pendientes.

## Evaluación y evolución experimental — 2026-11-26

Implementados probes de generación versionados y ciclo hipótesis/control/experimento/comparación. Ejecución real CPU: dos candidatos, diez probes cada uno, 0 aciertos y propuesta rechazada. No hay mejora de capacidades demostrada ni promoción automática. Ver EVOLUTION.md e informes JSON.

## Validación GPU completada — 2026-11-26

Entrenamiento, reanudación e inferencia propios verificados en RTX 4070 SUPER con FP32/FP16/BF16. Suite del entorno CUDA: 139 aprobadas, 4 omitidas (PostgreSQL). Checkpoint y métricas reales conservados. No se atribuye calidad conversacional. Comandos y límites: [CUDA_VALIDATION.md](CUDA_VALIDATION.md).

## Primer trabajo Foundation en nodos — 2026-11-26

Integrada tokenización BPE pública en scheduler/NodeAgent con capacidades declaradas, límites e integridad verificada. Dos procesos worker completaron dos shards por HTTP real en loopback. No equivale a entrenamiento neuronal distribuido; tampoco hay créditos ni verificación entre equipos todavía. Ver DISTRIBUTED_TRAINING.md.

## Entrenamiento neuronal con nodos — 2026-11-26

Integrado foundation.gradient.v1 con scheduler/NodeAgent y verificación por recomputación. Dos identidades de nodo aportan gradientes reales sobre shards públicos; agregación por tokens y AdamW producen checkpoints e inferencia. Demostración de dos rondas por HTTP loopback; solo un equipo y CPU. Créditos, multihost y escalabilidad de verificación pendientes. Ver DISTRIBUTED_TRAINING.md.
