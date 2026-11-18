#!/usr/bin/env bash
# Inciso 3: version CUDA (kernels escritos a mano con Numba CUDA).
#   1) comprueba que hay GPU y que Numba la reconoce,
#   2) verifica que los kernels dan los mismos resultados que PyTorch (referencia en CPU),
#   3) ejecuta REPS veces notebooks/asl_cuda.ipynb con el monitor de recursos en paralelo.
# Resultados: resultados/crudo/cuda_rep*.json
source "$(dirname "$0")/comun.sh"

banner "INCISO 3: version CUDA | REPS=$REPS EPOCHS=$EPOCHS WORKERS=$WORKERS (0 = todos los nucleos)"
"$PY" "$(ruta "$PRUEBAS_DIR/entorno.py")" 2>&1 | tee "$LOGS/entorno_cuda.log"
if ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['cuda_disponible'] else 1)" "$(ruta "$RESULTADOS/entorno.json")"; then
    echo "ERROR: Numba no ve ninguna GPU CUDA. Revisa el driver NVIDIA (nvidia-smi) y:  pip install \"numba-cuda[cu12]\" \"numpy<2.5\"" >&2
    exit 1
fi

if [ "${VERIFICAR:-1}" = "1" ]; then
    banner "3.1 Verificacion numerica de los kernels vs PyTorch (CPU)"
    "$PY" "$(ruta "$PRUEBAS_DIR/verificar_kernels.py")" 2>&1 | tee "$LOGS/verificacion_kernels.log"
else
    echo ">> (VERIFICAR=0: se omite la verificacion numerica de los kernels)"
fi

banner "3.2 Benchmark CUDA"
rm -f "$CRUDO"/cuda_rep*.json "$MONITOR"/cuda_rep*.csv
echo ">> Calentando el cache de disco (no se mide)..."
calentar_cache

for ((r = 1; r <= REPS; r++)); do
    export P3_REP="$r"
    echo; echo ">> Repeticion $r de $REPS   (captura el Administrador de Tareas y 'nvidia-smi -l 1' durante el entrenamiento)"
    iniciar_monitor "$MONITOR/cuda_rep$r.csv"
    ejecutar_nb "$NB_DIR/asl_cuda.ipynb" "$NB_EJEC/cuda_rep$r.ipynb" 2>&1 | tee "$LOGS/cuda_rep$r.log"
    detener_monitor
done
echo; echo "Listo. JSON por repeticion en: $CRUDO"
