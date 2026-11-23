# Registro de decisiones

Las decisiones aquí registradas son iniciales y pueden revisarse con evidencia.

## D-001 — Monolito modular para el primer backend

- **Estado:** aceptada.
- **Decisión:** comenzar con una aplicación FastAPI modular, en lugar de
  desplegar desde el inicio cada dominio como microservicio.
- **Motivo:** aún no hay carga medida ni contratos estables; un despliegue más
  simple reduce fallos operativos y permite endurecer límites de dominio.
- **Consecuencia:** definir interfaces de módulos y mantener la posibilidad de
  extraer componentes más adelante.

## D-002 — PostgreSQL como fuente de verdad transaccional

- **Estado:** aceptada.
- **Decisión:** usar PostgreSQL para identidades, estados de trabajo, auditoría
  y libro mayor de créditos. Redis, si se incorpora, no será autoridad de
  permisos ni saldo.
- **Motivo:** créditos y transiciones necesitan restricciones y transacciones
  duraderas.
- **Alcance actual:** el núcleo y la migración inicial se probaron con SQLite;
  falta validar el runtime y las transacciones contra PostgreSQL.

## D-003 — Nodos voluntarios como entornos no confiables

- **Estado:** aceptada.
- **Decisión:** no distribuir a nodos comunitarios conversaciones, memoria
  personal, secretos ni tareas con datos privados.
- **Motivo:** el propietario de un nodo puede inspeccionar el equipo y los
  procesos locales.
- **Consecuencia:** exigir clasificación de datos y política de elegibilidad
  antes de asignar cualquier trabajo.

## D-004 — No hay promoción automática de datos privados

- **Estado:** aceptada.
- **Decisión:** el consentimiento para conversar o guardar memoria no concede
  consentimiento para indexar globalmente ni entrenar.
- **Consecuencia:** consentimiento granular, trazable y revocable para cada
  propósito.

## D-005 — Apache-2.0 para el código del repositorio

- **Estado:** aceptada; aplicada en `LICENSE`.
- **Decisión:** aplicar Apache License 2.0 al código del proyecto.
- **Motivo:** licencia permisiva con concesión expresa de patentes, adecuada
  para contribuciones de software comunitarias.
- **Alcance:** no determina licencia ni permiso de redistribución para pesos de
  modelos, datasets, contenido web, dependencias o marcas. Revisarlos
  individualmente. Esta licencia tampoco crea una entidad legal sin ánimo de
  lucro.
