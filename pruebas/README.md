# Proyecto 3 — Pruebas y comparación CUDA vs secuencial

Todo se ejecuta con **scripts bash** (igual que en el Proyecto 2). Los notebooks son el código; los `.sh` los corren,
miden y comparan.

| Archivo | Qué hace |
|---|---|
| `notebooks/asl_lab3.ipynb` | Línea base **secuencial (solo CPU, sin CUDA)**. Secciones 12–13: análisis y benchmark. |
| `notebooks/asl_cuda.ipynb` | Versión **CUDA** con kernels escritos a mano (Numba CUDA, sin PyTorch) + comparación. |
| `src/data_utils.py` | Preparación de datos: `build_subsample` (secuencial) y `build_subsample_parallel` (CPU multinúcleo). |
| `src/cuda_kernels.py`, `src/cuda_cnn.py` | Los 13 kernels CUDA y la `SmallCNN` que los orquesta. |
| `src/bench_utils.py` | Cronometraje por fase y parámetros compartidos (variables `P3_*`). |
| `pruebas/inciso0_entorno.sh` | Hardware y versiones que reconoce el entorno (CPU, GPU, Numba). |
| `pruebas/inciso1_analisis_secuencial.sh` | Tablas de operaciones, repeticiones y tiempos medidos del programa secuencial. |
| `pruebas/inciso2_secuencial.sh` | Benchmark secuencial (REPS repeticiones, con monitor de recursos). |
| `pruebas/inciso3_cuda.sh` | Verifica los kernels contra PyTorch y corre el benchmark CUDA (REPS repeticiones, con monitor). |
| `pruebas/inciso4_comparacion.sh` | Tablas de tiempos, **speedup por fase**, uso de recursos y gráficos. |
| `pruebas/run_all.sh` | Todo lo anterior en orden. |

## Preparar el entorno (Windows con GPU NVIDIA)

Necesitas el **driver NVIDIA** (`nvidia-smi` debe funcionar) y **Git Bash** o **WSL**. Desde la raíz del repo:

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash.  (WSL/Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt      # incluye numba-cuda[cu12] y numpy<2.5
```

> **NumPy debe ser < 2.5.** Con NumPy 2.5 Numba CUDA falla al compilar el primer kernel (`numpy.row_stack` fue eliminado).
> Por eso `requirements.txt` fija `numpy==2.4.6`.

El dataset va en `data/raw/asl_alphabet_train/asl_alphabet_train/<clase>/*.jpg` (como en el Lab 3). Si lo tienes en otra
carpeta: `RAW_DIR=/ruta/al/asl_alphabet_train bash pruebas/run_all.sh`.

## Comandos

```bash
bash pruebas/run_all.sh                       # todo (REPS=3, EPOCHS=15); en la máquina con GPU

# o paso a paso
bash pruebas/inciso0_entorno.sh               # evidencia: CPU/GPU reconocidos
bash pruebas/inciso1_analisis_secuencial.sh   # análisis de la solución secuencial
bash pruebas/inciso2_secuencial.sh            # benchmark secuencial
bash pruebas/inciso3_cuda.sh                  # verificación de kernels + benchmark CUDA
bash pruebas/inciso4_comparacion.sh           # tablas y gráficos de la comparación
```

Prueba rápida (humo): `REPS=1 EPOCHS=2 bash pruebas/run_all.sh`.

Variables opcionales: `REPS` (3) · `EPOCHS` (15) · `BATCH` (128) · `LR` (1e-3) · `WD` (1e-4) · `N_PER_CLASS` (600) ·
`IMG_SIZE` (64) · `SEED` (42) · `WORKERS` (0 = todos los núcleos lógicos, procesos de preparación en la versión CUDA) ·
`SEQ_THREADS` (1) · `PYTHON` (intérprete a usar) · `VERIFICAR=0` (omite la verificación de kernels).

## Condiciones de la comparación (para que el speedup sea honesto)

* **Corre secuencial y CUDA en la misma máquina** (la de la GPU). Los tiempos de otra máquina (p. ej. la Mac) no sirven para el speedup.
* Ambas versiones usan **los mismos datos** (preparación idéntica, verificado en el notebook CUDA), **la misma arquitectura**
  (`SmallCNN`, 1,057,533 parámetros), mismo batch, lr, weight decay, épocas y semilla.
* **Secuencial = 1 hilo** (`SEQ_THREADS=1`, `OMP_NUM_THREADS=1`) y 1 proceso para preparar imágenes. Para ver una línea base más
  fuerte: `SEQ_THREADS=0` usa los hilos por defecto de PyTorch; reporta ambos si lo haces.
* La **compilación JIT** de los kernels se mide aparte y no entra en los totales; el tiempo CUDA **sí incluye** la transferencia CPU→GPU y GPU→CPU.
* Antes de medir se "calienta" el caché de disco para que ninguna versión pague el disco frío.
* Cada paso hace `REPS` repeticiones y reporta media ± desviación estándar.

## Dónde queda cada evidencia (rúbrica)

| Rubro | Archivos en `pruebas/resultados/` |
|---|---|
| Refactorización: CPU para imágenes, CUDA solo para la red | `src/data_utils.py` (CPU multinúcleo), `src/cuda_*.py` (GPU), notebook CUDA secciones 3–5 |
| Logs de que se usó el hardware | `logs/entorno.log`, `logs/entorno_cuda.log`, `logs/verificacion_kernels.log`, `logs/*_rep*.log`, `notebooks_ejecutados/*.ipynb` (incluye `nvidia-smi`) |
| Uso de recursos por núcleo / GPU | `monitor/*.csv`, `figuras/recursos_secuencial.png`, `figuras/recursos_cuda.png`, y la Tabla 5/6 de `tablas.md` |
| Tabla comparativa y speedup por fase | `tablas.md`, `tiempos_por_fase.csv`, `figuras/tiempos.png`, `figuras/speedup.png`, `figuras/epocas.png` |
| Análisis de la solución secuencial | `perfil_secuencial.md` |

### Capturas de pantalla del informe

Las figuras `recursos_*.png` salen del monitor automático, pero el enunciado pide además capturas del sistema. Mientras corre
`inciso2_secuencial.sh` / `inciso3_cuda.sh`:

1. **Administrador de Tareas → Rendimiento → CPU**, clic derecho → *Cambiar gráfico a → Procesadores lógicos* (se ve cada núcleo).
   Captura durante la preparación de datos (versión CUDA: varios núcleos al 100 %; secuencial: uno solo).
2. **Administrador de Tareas → Rendimiento → GPU** (Cuda / memoria dedicada) y, en otra terminal, `nvidia-smi -l 1`.
   Captura durante el entrenamiento CUDA.
3. Si el entrenamiento CUDA termina demasiado rápido para capturarlo, alarga la corrida: `EPOCHS=60 REPS=1 bash pruebas/inciso3_cuda.sh`
   (solo para las capturas; la comparación oficial usa las mismas épocas en ambas versiones).
