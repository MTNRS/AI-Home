# Validación CUDA — 2026-11-26

GPU verificada: NVIDIA GeForce RTX 4070 SUPER, 12.878.086.144 bytes de VRAM,
compute capability 8.9; controlador observado por nvidia-smi 610.88.
Entorno aislado `.venv-cuda`: Python 3.11.9 y PyTorch 2.14.1+cu130,
runtime CUDA 13.0. BF16 disponible. Inventario en CUDA_RUNTIME.json.

La instalación Python habitual conserva PyTorch 2.15.0+cpu. El venv usa las
dependencias generales instaladas mediante system-site-packages; no es un
lockfile completamente aislado. La variante CUDA se instaló desde el índice
oficial de PyTorch, cuya disponibilidad se comprobó con pip index versions.
Guía del proveedor: [instalación de PyTorch](https://pytorch.org/get-started/locally/).

## Comandos ejecutados

```powershell
python -m venv --system-site-packages .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.\.venv-cuda\Scripts\python.exe -m pytest services/api/tests/test_foundation_model.py -q
.\.venv-cuda\Scripts\python.exe -m pytest -q
$env:PYTHONPATH="services/api/src"
.\.venv-cuda\Scripts\python.exe -m ai_home.foundation.smoke --output-dir artifacts/foundation-smoke-cuda --device cuda
```

Pruebas del modelo: 8 aprobadas. Suite completa: **139 aprobadas, 4 omitidas**,
17,36 segundos; solo se omiten pruebas PostgreSQL sin servidor configurado.
Causalidad, RoPE, GQA/KV cache y gradientes se prueban en CPU y CUDA.
FP32, FP16 y BF16 se entrenan con acumulación y gradient checkpointing, se
reanuda el optimizador y se comparan pesos dentro de tolerancia 1e-6/1e-5.
La prueba también carga y genera en CUDA. Esto no garantiza reproducibilidad
bit a bit entre hardware, drivers o versiones distintas.

Smoke real: ocho pasos, checkpoint intermedio en paso cuatro, reanudación y
carga final en cuda:0. 27.808 parámetros; cambio L2 de embeddings tras resume
0,02364743; pérdida de validación 5,615622. Pico del asignador PyTorch
17.749.504 bytes: excluye contexto CUDA, otros procesos y memoria del driver.
La muestra es deliberadamente diminuta y no mide el rendimiento máximo de la GPU.
La generación permanece incoherente; no acredita capacidades conversacionales.

Informe conservado en FOUNDATION_SMOKE_CUDA.json; dataset y checkpoints reales
en artifacts/foundation-smoke-cuda, excluidos de Git. Las mediciones CPU previas
usaron otra versión/estado del empaquetado y no deben interpretarse como una
comparación controlada CPU/GPU. Para perfilar tamaños mayores hacen falta
benchmarks propios, presupuestos y conjuntos de evaluación representativos.

Distribución entre nodos, créditos por entrenamiento y perfiles automáticos de
memoria siguen pendientes. Se inspeccionaron los handlers existentes: solo
math.sum.v1 está integrado en el scheduler, aún no tareas Foundation.
