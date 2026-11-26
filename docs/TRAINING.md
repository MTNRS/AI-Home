# Entrenamiento reproducible

Requiere Python >=3.11, PyTorch >=2.6 y dependencias del proyecto. Instalar con python -m pip install -e ".[dev,llm]" en un entorno con acceso a paquetes. Para CUDA, instalar una distribución de PyTorch compatible con el equipo.

Prueba mecánica completa preparada:

```powershell
python -m ai_home.foundation.smoke --output-dir artifacts/foundation-smoke --device cpu
```

El comando está diseñado para crear un corpus sintético declarado, aprender BPE, inicializar pesos aleatorios, entrenar cuatro pasos, cargar el optimizador y continuar hasta ocho, guardar checkpoints y generar texto real. Es un experimento de mecánica, no un benchmark de calidad. Completado en CPU el 2026-11-26; resultados conservados en `docs/FOUNDATION_SMOKE_CPU.json`. Los checkpoints y el dataset permanecen en `artifacts/foundation-smoke-cpu/` (excluidos de Git). Si el paquete no está instalado, definir `$env:PYTHONPATH="services/api/src"` antes del comando.

Entrenamiento sobre datos propios:

```powershell
python -m ai_home.foundation --dataset dataset.json --output foundation-nano.pt --device auto
```

El dataset es JSON con format_version=1 y records. Cada registro requiere source, url, license, date, language, text, training_allowed=true, privacy_reviewed=true y visibility=public. Los permisos son declaraciones revisadas por el operador, no una verificación jurídica automática. El filtrado detecta patrones comunes de secretos y correos, no todos los datos personales. No conectar directamente conversaciones privadas.

Se separan documentos distintos antes de entrenar el tokenizador. Las secuencias no cruzan documentos; las colas se rellenan con PAD y sus etiquetas se excluyen de la pérdida. La partición se ordena por hash y reserva 20%: añadir documentos puede cambiarla. No sirve todavía como evaluación estable de aprendizaje continuo. Deduplicación semántica y por familias documentales pendiente.

AdamW, warmup/coseno, acumulación, clipping, fp32/fp16/bf16, detección de valores no finitos, métricas y checkpoints periódicos. Resume exige dataset y configuración idénticos y conserva estados del optimizador, escalador y RNG. Usa --stop-after para interrumpir sin modificar el horizonte del scheduler. Cambiar de dispositivo no garantiza reproducibilidad bit a bit.

Las cargas usan weights_only=True, comprobación de dimensiones, finitud y huella del tokenizador. Esto reduce riesgos, pero no convierte checkpoints desconocidos en archivos confiables ni evita todos los agotamientos de recursos. Solo cargar artefactos de origen aprobado. Los archivos de checkpoint no deben añadirse directamente a Git: publicar posteriormente con procedencia y revisión.


## Contribuciones revisadas y SFT

Exportar un dataset congelado desde la API administrativa existente; conservar
el archivo en almacenamiento del operador. El hash comprueba integridad, no
identidad del emisor: solo usar exportaciones de la instancia autorizada.

Crear una configuración JSON de entrenamiento, por ejemplo:
`{"steps": 20, "warmup": 2, "objective": "sft"}` y ejecutar:

```powershell
python -m ai_home.foundation --reviewed-export reviewed.json --training-config sft.json --output artifacts/sft-candidate.pt --device cpu
```

El importador comprueba versión y hash del manifiesto, número de muestras,
consentimiento, revisión humana completa, roles alternos y filtros de contenido.
El consentimiento no autoriza redistribución: los registros y pesos resultantes
se etiquetan `reviewed-training-only`; no se envían a nodos ni se publican.
No se incorpora ningún mensaje directamente desde el chat. Los filtros comunes
no sustituyen la revisión humana ni detectan todos los datos personales.

SFT usa el mismo formato de roles que inferencia. Solo contenido del asistente
y su EOS contribuyen a la pérdida; roles, preguntas y PAD quedan enmascarados.
Las conversaciones se separan antes de entrenar BPE. El checkpoint conserva
versión/hash del dataset, objetivo, métricas y estado candidato, sin almacenar
el texto de las conversaciones. Los pesos pueden memorizar datos: borrar la
exportación no elimina su influencia; una retirada posterior exige reconstruir
versiones afectadas o investigar unlearning, que aún no está implementado.

Actualmente este comando inicializa un modelo propio desde cero o reanuda el
mismo entrenamiento. Fine-tuning desde otro candidato, replay, evaluación de
regresión estable y promoción aprobada siguen pendientes. Esta ruta demuestra
SFT mecánico, no capacidad conversacional. El empaquetado ahora incluye colas:
no comparar métricas históricas como si fueran la misma configuración de datos.
La nueva configuración incluye `objective`; resume rechaza configuraciones
antiguas sin ese campo en lugar de asumir equivalencia.
