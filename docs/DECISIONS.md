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

## Foundation Model — implementación inicial (2026-11-26)

Se añade un subsistema propio en `ai_home.foundation` con BPE desde cero, Transformer RoPE/GQA, entrenamiento, checkpoints e inferencia. El modelo byte-level y las funcionalidades anteriores se conservan. La integración de escritorio añade un proveedor independiente para checkpoints propios.

**Estado: parcial y bloqueado para validación neuronal.** Cuatro pruebas sin PyTorch pasan; no se ha completado entrenamiento, generado un checkpoint ni ejecutado inferencia nueva. PyTorch/pytest no están disponibles y su instalación falla. La escritura de ramas en GitHub devuelve 403 y Git local no puede crear `.git/index.lock`; no se han creado commits.

Consultar [auditoría y matriz completa](FOUNDATION_AUDIT.md), [modelo](FOUNDATION_MODEL.md), [entrenamiento](TRAINING.md) y [pruebas](TESTING.md). Aprendizaje continuo, distribución, créditos de entrenamiento, RAG público, evolución y paneles completos siguen pendientes; no se reduce el objetivo ni se declaran demostrados.



## 2026-11-26: backend Foundation explícito y aislamiento por petición

Foundation pasa a ser el backend HTTP predeterminado. El proxy externo se conserva con selección explícita. No se comparte KV cache ni se ingieren mensajes del chat; el checkpoint es configuración del servidor. El bloqueo no espera ni acumula peticiones: responde ocupado. El límite es por proceso, no un scheduler global.

## 2026-11-26: consentimiento de entrenamiento no implica publicación

Las exportaciones revisadas entran por una opción explícita, separada del corpus público. Se mantienen como reviewed-training-only y SFT enmascara prompts. Se conserva manifiesto en el checkpoint y no se añade ninguna distribución a nodos. El hash no autentica el archivo: el operador debe obtenerlo por la API administrativa autorizada.

## 2026-11-26: separar reanudación y aprendizaje incremental

Resume restaura el optimizador y exige datos/configuración idénticos. Inicializar un candidato conserva pesos y tokenizador propios pero crea un optimizador nuevo. Los hashes de entrenamiento y holdout se acumulan por linaje; un checkpoint sin ese historial no puede demostrar aislamiento y se rechaza como padre incremental. Un resultado favorable de pérdida nunca activa publicación.

## 2026-11-26: experimento rechazado también es evidencia

El primer ciclo compara tasa de aprendizaje con un control del mismo presupuesto. Requiere ganancia de acierto y ausencia de regresiones; una pérdida menor aislada no basta. El resultado real fue rechazo y se conserva. Diez probes autorados no se presentan como benchmark general ni prueba de capacidades adquiridas.

## 2026-11-26: capacidad declarada y artefactos verificados en nodos

Clientes antiguos conservan solo suma. Los nuevos anuncian tokenización explícitamente; el servidor no asigna tareas por suponer que todo nodo soporta Foundation. Se reejecuta el trabajo acotado para verificar integridad, aceptando su coste duplicado hasta disponer de un protocolo independiente más escalable. No se remuneran tareas autodeclaradas ni se reinterpretan permisos de conversaciones.
