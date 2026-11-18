#!/usr/bin/env bash
# Inciso 2: medicion del tiempo de ejecucion SECUENCIAL.
# Ejecuta REPS veces la seccion de benchmark de notebooks/asl_lab3.ipynb (solo CPU, sin CUDA),
# con el monitor de recursos en paralelo. Resultados: resultados/crudo/secuencial_rep*.json
source "$(dirname "$0")/comun.sh"

banner "INCISO 2: benchmark SECUENCIAL | REPS=$REPS EPOCHS=$EPOCHS SEQ_THREADS=$SEQ_THREADS"
rm -f "$CRUDO"/secuencial_rep*.json "$MONITOR"/secuencial_rep*.csv      # descarta corridas anteriores
export OMP_NUM_THREADS="$SEQ_THREADS" MKL_NUM_THREADS="$SEQ_THREADS" OPENBLAS_NUM_THREADS="$SEQ_THREADS"

echo ">> Calentando el cache de disco (no se mide)..."
calentar_cache

for ((r = 1; r <= REPS; r++)); do
    export P3_REP="$r"
    echo; echo ">> Repeticion $r de $REPS   (abre el Administrador de Tareas / Monitor de actividad ahora si quieres capturas)"
    iniciar_monitor "$MONITOR/secuencial_rep$r.csv"
    ejecutar_nb "$NB_DIR/asl_lab3.ipynb" "$NB_EJEC/secuencial_rep$r.ipynb" --solo-tag benchmark 2>&1 | tee "$LOGS/secuencial_rep$r.log"
    detener_monitor
done
echo; echo "Listo. JSON por repeticion en: $CRUDO"
