#!/usr/bin/env bash
# Inciso 0: deja constancia de que el entorno reconoce el hardware (CPU, GPU, versiones).
# Evidencia para el rubro "Evidencia de ejecucion": logs/entorno.log y entorno.json.
source "$(dirname "$0")/comun.sh"

banner "INCISO 0: entorno y hardware"
{
    date
    echo "Python: $("$PY" --version 2>&1)  ($PY)"
    echo "--- paquetes relevantes ---"
    "$PY" -m pip list 2>/dev/null | grep -i -E "^(torch|numpy|numba|numba-cuda|cuda-|nvidia-|psutil|pandas|matplotlib|nbclient|ipykernel|pillow) " || true
    echo "--- entorno (CPU, PyTorch, Numba CUDA, GPU) ---"
    "$PY" "$(ruta "$PRUEBAS_DIR/entorno.py")"
    echo "--- nvidia-smi -L ---"
    nvidia-smi -L 2>&1 || echo "(nvidia-smi no disponible)"
} 2>&1 | tee "$LOGS/entorno.log"
