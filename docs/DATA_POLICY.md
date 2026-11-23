# Política inicial de datos

Esta política describe requisitos de diseño. No afirma que los controles ya
estén implementados ni sustituye una revisión legal antes del despliegue.

## Clases y tratamiento

| Clase | Ejemplos | Tratamiento previsto |
|---|---|---|
| Pública autorizada | Documentación abierta, datasets con licencia compatible | Registrar fuente, licencia, propósito y fecha; permitir retirada cuando corresponda |
| Operativa | Estado de trabajos, métricas mínimas | Minimizar, limitar retención y restringir acceso |
| Personal privada | Conversaciones, archivos, preferencias, memoria | Aislar por cuenta; no indexar globalmente ni enviar a nodos voluntarios |
| Secreta | Contraseñas, tokens, claves | No registrar ni enviar a nodos; almacenar con mecanismos adecuados para cada secreto |

## Consentimiento y propósito

- La recopilación debe tener un propósito explícito y minimizado.
- Consentir una función no implica consentir otros usos.
- Las contribuciones a evaluación, conocimiento global y entrenamiento son
  opciones separadas.
- Registrar la versión del consentimiento y facilitar su revocación y la
  solicitud de eliminación.
- No incluir por defecto contenido privado en telemetría, prompts de terceros,
  benchmarks públicos o datasets.

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
ser inferida o memorizada.

## Incidentes

El sistema deberá contar con un procedimiento de contención, evaluación,
notificación y remediación de exposiciones de datos antes de admitir datos
personales en un despliegue de producción.
