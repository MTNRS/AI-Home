# Política inicial de datos

Esta política describe los requisitos de diseño y algunos controles
experimentales ya implementados. No sustituye una revisión legal antes del
despliegue.

## Clases y tratamiento

| Clase | Ejemplos | Tratamiento previsto |
|---|---|---|
| Pública autorizada | Documentación abierta, datasets con licencia compatible | Registrar fuente, licencia, propósito y fecha; permitir retirada cuando corresponda |
| Operativa | Estado de trabajos, métricas mínimas | Minimizar, limitar retención y restringir acceso |
| Personal privada | Conversaciones, archivos, preferencias, memoria | Aislar por cuenta; no indexar globalmente ni enviar a nodos voluntarios |
| Secreta | Contraseñas, tokens, claves | No registrar ni enviar a nodos; almacenar con mecanismos adecuados para cada secreto |

## Consentimiento y propósito

- El proveedor de identidad configurado (Supabase Auth) procesa credenciales,
  direcciones de correo, sesiones y los flujos de verificación de cuenta. La
  aplicación no guarda contraseñas de cuentas gestionadas por Supabase.
- FastAPI comprueba los access tokens de Supabase mediante las claves públicas
  JWKS del mismo proyecto y conserva el UUID/correo mínimo necesario para
  relacionar recursos. La sesión revocable del API almacena únicamente el hash
  del access token, no el token original.
- La clave `SUPABASE_SECRET_KEY` se limita al backend y solo se usa para borrar
  una identidad cuando la persona solicita eliminar su cuenta. La clave pública
  publishable no sustituye controles de acceso en la API.
- La recopilación debe tener un propósito explícito y minimizado.
- Consentir una función no implica consentir otros usos.
- Las contribuciones a evaluación, conocimiento global y entrenamiento son
  opciones separadas.
- Registrar la versión del consentimiento y facilitar su revocación y la
  solicitud de eliminación.
- No incluir por defecto contenido privado en telemetría, prompts de terceros,
  benchmarks públicos o datasets.
- La aplicación de escritorio mantiene el historial de chat en memoria. Solo
  al marcar el opt-in por conversación y pulsar la acción de aporte se envía el
  historial completo al servidor AI@Home autenticado para guardarlo como
  candidato privado pendiente de revisión.
- Antes de enviar, la interfaz informa que el operador del servidor puede
  acceder a esos datos y que no se garantiza deshacer su influencia en pesos
  ya publicados. No se debe aportar información sensible.
- El usuario debe revisar el contenido y confirmar por separado que las
  condiciones del proveedor permiten reutilizarlo. Usar BYOK por sí solo no
  constituye consentimiento ni una licencia para entrenamiento.

## Memoria personal local

Las notas personales se guardan como archivos Markdown en la carpeta de
configuración local del usuario (`ai-home/memory`) y no se sincronizan con el
servidor. El usuario puede crearlas,
enlazarlas y eliminarlas desde el cliente de escritorio; los enlaces entre notas
son compatibles con el grafo de Obsidian. La búsqueda de recuerdos se ejecuta
localmente en cada pregunta: solo las notas coincidentes se añaden al contexto
de la solicitud y se envían al proveedor seleccionado (también al servidor
AI@Home si se eligió ese modo). El uso como contexto de inferencia no entrena ni
modifica los pesos del modelo. El contenido Markdown no está cifrado por la
aplicación: su protección en reposo depende de la cuenta y el dispositivo local.

## Conversaciones candidatas a entrenamiento

La API requiere autenticación, consentimiento afirmativo, versión y alcance
fijos, procedencia de proveedor/modelo, revisión humana declarada y
confirmación de derechos de reutilización. Cada conversación tiene un
identificador único por cuenta y solo puede enviarse una vez. Los registros se
marcan `pending_review`; no se promocionan ni se usan para entrenar
automáticamente. Un filtro adicional bloquea algunos patrones comunes de
credenciales en los mensajes y endpoints, pero no detecta de forma fiable
información personal ni todos los secretos.

Las candidaturas se guardan como JSON en la base de datos configurada por el
operador; esta versión no cifra el contenido de la aplicación a nivel de
campo. El operador de la instancia y cualquier persona con acceso autorizado
a la base de datos pueden leerlas. El cliente exige HTTPS para servidores
remotos. La persona puede consultar y borrar sus candidaturas antes de que
formen parte de un dataset congelado; borrar la cuenta también las elimina.
La revisión y congelación están disponibles mediante API administrativa
protegida por el secreto `AI_HOME_TRAINING_REVIEW_TOKEN`, que debe ser aleatorio,
tener al menos 32 caracteres y mantenerse fuera del repositorio. La persona
operadora configura también `AI_HOME_TRAINING_REVIEWER_ID`; se registra ese
identificador, no se confía en un valor enviado por quien llama. Aprobar exige
confirmar individualmente revisión de términos, datos sensibles y derechos de
reutilización. Rechazar vacía el contenido de la candidatura. El listado de
revisión puede exponer el contenido completo a quien posea el token.

La congelación solo acepta candidaturas aprobadas, guarda un snapshot con
procedencia y checklist de revisión, asigna una versión única e inmutable y
calcula SHA-256 sobre el manifiesto canónico. La retirada de una candidatura
frozen se rechaza; borrar la cuenta elimina el registro de contribución, pero
no el snapshot congelado. La influencia sobre pesos publicados no puede
garantizarse reversible. Aún no hay entrenamiento implementado que consuma
datasets ni interfaz administrativa para la revisión.

## Fuentes externas y procedencia

Antes de adquirir contenido se debe determinar si el acceso y el uso previsto
están permitidos. Registrar, como mínimo, URL o identificador de origen,
titular/fuente cuando se conozca, licencia o base de permiso, fecha de acceso,
transformaciones y restricciones.

No eludir autenticación, paywalls, medidas técnicas ni límites de acceso.
Respetar límites de carga y mecanismos de exclusión aplicables. El hecho de
que un recurso sea visible públicamente no basta para justificar su descarga,
redistribución o uso en entrenamiento.

## Retención, acceso y eliminación

La implementación deberá definir plazos de retención por categoría, controles
de acceso por recurso, exportación y eliminación verificable. Copias de
seguridad, índices, embeddings, cachés, logs y derivados deben incluirse en el
análisis de eliminación. No se promete borrado instantáneo de copias de
seguridad antes de definir y publicar sus plazos.

## Entrenamiento

Solo se usarán datos con autorización documentada para ese propósito. Mantener
manifest de datasets, versiones, licencias, filtros y lineage del modelo. La
anonimización simple no se considera garantía de que una contribución no pueda
ser inferida o memorizada. Las candidaturas actuales son únicamente una etapa
de recolección/revisión; los snapshots versionados y con hash quedan congelados
para una futura evaluación. Falta integrar entrenamiento fundacional real y
verificar legalmente las condiciones de cada proveedor/dataset.

## Claves de proveedores

El cliente almacena claves BYOK solo en el almacén de credenciales del sistema
operativo. Las solicitudes directas envían la clave en el encabezado
autorizado al endpoint del proveedor seleccionado; no se guardan en la
configuración JSON ni se envían a AI@Home. Los adaptadores nativos fijan el
endpoint oficial de cada proveedor. Los endpoints personalizados remotos
requieren HTTPS y la persona debe confiar en su operador. Si falla el almacén
seguro, el cliente no recurre a almacenamiento en texto plano.

## Incidentes

El sistema deberá contar con un procedimiento de contención, evaluación,
notificación y remediación de exposiciones de datos antes de admitir datos
personales en un despliegue de producción.
