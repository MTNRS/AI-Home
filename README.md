# AI@Home

AI@Home es un proyecto comunitario de inteligencia artificial abierta, con el
objetivo de permitir investigación colaborativa y computación voluntaria sin
comprometer la privacidad de sus participantes.

El proyecto está en su fase inicial. Este repositorio contiene las decisiones
de arquitectura y las políticas de base; todavía no ofrece una API, un nodo de
cómputo ni un sistema de créditos funcionales.

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

## Dirección técnica inicial

El primer backend se implementará como un **monolito modular** en Python con
FastAPI. Esta decisión reduce la complejidad operativa mientras los límites
entre componentes todavía evolucionan. PostgreSQL será la fuente de verdad
para usuarios, trabajos y el libro mayor de créditos; Redis podrá utilizarse
para coordinación y colas, nunca como libro mayor.

La distribución de trabajos se habilitará gradualmente. No se enviará memoria
personal, conversaciones ni otros datos privados a nodos voluntarios.

## Desarrollo

La documentación actual es una base de diseño, no una implementación
ejecutable. Consulta [PROGRESS.md](docs/PROGRESS.md) para conocer qué existe y
qué no. Las decisiones que afecten a privacidad, confianza, créditos o
publicación de modelos deben quedar registradas antes de desplegar esas
funciones.

## Estado del proyecto

El repositorio de GitHub es actualmente privado. La intención de código abierto
no sustituye la elección y aplicación de una licencia: la publicación y la
licencia deberán decidirse antes de una primera versión pública.
