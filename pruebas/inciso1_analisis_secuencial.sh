#!/usr/bin/env bash
# Inciso 1: analisis de la solucion secuencial. Mide en que operaciones se va el tiempo y
# cuantas veces se repite cada una (tablas del documento: "Operaciones identificadas" y
# "Operaciones repetitivas"). Salida: resultados/perfil_secuencial.md y .csv
source "$(dirname "$0")/comun.sh"

banner "INCISO 1: analisis y perfil de la solucion secuencial (CPU, $SEQ_THREADS hilo/s)"
export OMP_NUM_THREADS="$SEQ_THREADS" MKL_NUM_THREADS="$SEQ_THREADS" OPENBLAS_NUM_THREADS="$SEQ_THREADS"
"$PY" "$(ruta "$PRUEBAS_DIR/perfil_secuencial.py")" 2>&1 | tee "$LOGS/perfil_secuencial.log"
