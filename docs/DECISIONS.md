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
- **Alcance actual:** el primer núcleo probado usa SQLAlchemy y SQLite en
  memoria; PostgreSQL, migraciones y despliegue no están validados todavía.

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

## D-005 — Licencia pendiente de decisión

- **Estado:** pendiente.
- **Decisión:** no añadir una licencia por defecto. Elegir una licencia de
  software y definir licencias para documentación, modelos y datasets antes de
  abrir el repositorio al público.
- **Motivo:** “sin ánimo de lucro” y “código abierto” no determinan por sí
  solos los derechos de uso y redistribución.
