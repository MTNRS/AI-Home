# Modelo inicial de seguridad

Este documento define requisitos de ingeniería; no es una certificación ni
afirma que el producto tenga ya controles desplegados.

## Activos que proteger

- Credenciales, claves y sesiones.
- Conversaciones, archivos y memoria privada.
- Integridad de los saldos y del libro mayor.
- Integridad y disponibilidad de trabajos.
- Código, artefactos de modelos y cadena de suministro.
- Consentimientos, procedencia y registros de auditoría.

## Límites de confianza

- Clientes y dispositivos de usuario.
- Nodos voluntarios, que se consideran no confiables.
- API y servicios operados por el proyecto.
- Proveedores y fuentes externas.
- Artefactos generados por modelos o colaboradores.

## Amenazas prioritarias

- Acceso a datos entre cuentas por fallos de autorización.
- Repetición o manipulación de resultados para obtener créditos indebidos.
- Ejecución de trabajos que excedan los permisos acordados por el nodo.
- Robo de credenciales, secretos en logs o exposición por cachés/embeddings.
- Envenenamiento de datos, abuso de fuentes y contaminación de benchmarks.
- Dependencias, modelos o actualizaciones de cliente comprometidos.
- Cambios generados automáticamente que debiliten controles críticos.

## Requisitos antes de producción

1. Autorización por recurso, con pruebas negativas de acceso entre usuarios.
2. Secretos fuera del control de versiones y de logs.
3. TLS en tránsito y cifrado en reposo apropiado al despliegue, con gestión y
   rotación de claves definidas.
4. Ledger transaccional e idempotente, con controles de concurrencia y
   auditoría.
5. Validación de entradas, límites de tasa y tamaño, y límites de recursos.
6. Aislamiento de trabajos no confiables; sin acceso al host o a la red por
   defecto.
7. Dependencias y artefactos fijados/verificados; actualizaciones autenticadas.
8. Revisión humana de cambios en autenticación, privacidad, permisos y créditos.
9. Procedimiento de respuesta a incidentes, copias de seguridad y restauración
   probada.
10. Evaluación de riesgos de extracción y memorización antes de usar datos
    privados en cualquier entrenamiento.

## Reporte de vulnerabilidades

Antes de anunciar un programa público de recompensas, se habilitará un canal
privado de reporte y un procedimiento de triage. No enviar información
explotable de vulnerabilidades a issues públicos. Los detalles del canal se
publicarán cuando exista una persona/equipo responsable de atenderlos.
