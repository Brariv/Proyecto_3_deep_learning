#!/usr/bin/env bash
# Definiciones compartidas por los scripts de prueba del Proyecto 3 (secuencial vs CUDA).
# Cada script hace:  source "$(dirname "$0")/comun.sh"
# Funciona en macOS, Linux, WSL y Git Bash (Windows).
set -euo pipefail

PRUEBAS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ="$(cd "$PRUEBAS_DIR/.." && pwd)"
NB_DIR="$RAIZ/notebooks"
RESULTADOS="$PRUEBAS_DIR/resultados"
CRUDO="$RESULTADOS/crudo"            # un JSON por repeticion (tiempos por fase, historial, metadatos)
MONITOR="$RESULTADOS/monitor"        # CSV de CPU por nucleo / GPU durante cada repeticion
LOGS="$RESULTADOS/logs"              # salida de consola de cada paso
NB_EJEC="$RESULTADOS/notebooks_ejecutados"
mkdir -p "$CRUDO" "$MONITOR" "$LOGS" "$NB_EJEC"

# Ruta que entiende Python nativo de Windows (en Git Bash /c/Users/... -> C:/Users/...).
ruta() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }

# Elige el interprete: $PYTHON, el .venv del repo, python3, python o py.
elegir_python() {
    if [ -n "${PYTHON:-}" ]; then printf '%s' "$PYTHON"; return 0; fi
    local c
    for c in "$RAIZ/.venv/Scripts/python.exe" "$RAIZ/.venv/bin/python" python3 python py; do
        if { [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; } &&
           "$c" -c "import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)" >/dev/null 2>&1; then
            printf '%s' "$c"; return 0
        fi
    done
    echo "ERROR: no encontre Python 3. Activa tu entorno virtual o define PYTHON=/ruta/a/python" >&2
    return 1
}
PY="$(elegir_python)"

# ---- Parametros del experimento (se pueden sobreescribir:  EPOCHS=5 REPS=1 bash pruebas/run_all.sh) ----
REPS=${REPS:-3}                    # repeticiones por version (se reporta media y desviacion)
EPOCHS=${EPOCHS:-15}               # epocas de entrenamiento (Lab 3: 15)
BATCH=${BATCH:-128}
LR=${LR:-1e-3}
WD=${WD:-1e-4}
N_PER_CLASS=${N_PER_CLASS:-600}    # imagenes por clase (Lab 3: 600 -> 17,400 imagenes)
IMG_SIZE=${IMG_SIZE:-64}
SEED=${SEED:-42}
WORKERS=${WORKERS:-0}              # procesos CPU para preparar imagenes en la version CUDA (0 = todos los nucleos)
SEQ_THREADS=${SEQ_THREADS:-1}      # hilos de PyTorch en la linea base secuencial (1 = estrictamente secuencial)
INTERVALO_MON=${INTERVALO_MON:-0.5}

export P3_EPOCHS="$EPOCHS" P3_BATCH="$BATCH" P3_LR="$LR" P3_WD="$WD" P3_N_PER_CLASS="$N_PER_CLASS"
export P3_IMG_SIZE="$IMG_SIZE" P3_SEED="$SEED" P3_WORKERS="$WORKERS" P3_SEQ_THREADS="$SEQ_THREADS"
[ -n "${RAW_DIR:-}" ] && export P3_RAW_DIR="$(ruta "$RAW_DIR")"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

banner() { printf '\n==================== %s ====================\n' "$*"; }

# ---- Monitor de recursos en segundo plano (CPU por nucleo, RAM, GPU) ----
MON_PID=""; MON_STOP=""
iniciar_monitor() {   # $1 = CSV de salida
    MON_STOP="$1.stop"; rm -f "$MON_STOP"
    "$PY" "$(ruta "$PRUEBAS_DIR/monitor_recursos.py")" --salida "$(ruta "$1")" \
        --intervalo "$INTERVALO_MON" --parar "$(ruta "$MON_STOP")" &
    MON_PID=$!
}
detener_monitor() {
    if [ -n "$MON_PID" ]; then
        touch "$MON_STOP"; wait "$MON_PID" 2>/dev/null || true; rm -f "$MON_STOP"; MON_PID=""
    fi
}
trap detener_monitor EXIT

calentar_cache() { "$PY" "$(ruta "$PRUEBAS_DIR/calentar_cache.py")"; }

# ejecutar_nb <notebook.ipynb> <copia_ejecutada.ipynb> [opciones de ejecutar_notebook.py]
ejecutar_nb() {
    local entrada="$1" salida="$2"; shift 2
    "$PY" "$(ruta "$PRUEBAS_DIR/ejecutar_notebook.py")" "$(ruta "$entrada")" "$(ruta "$salida")" "$@"
}
