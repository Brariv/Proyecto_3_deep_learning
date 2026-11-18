#!/usr/bin/env bash
# Corre TODO el Proyecto 3 en orden: entorno -> analisis secuencial -> benchmark secuencial
# -> version CUDA -> comparacion. Pensado para ejecutarse en la maquina con GPU NVIDIA.
#
#   bash pruebas/run_all.sh                 # corrida completa (REPS=3, EPOCHS=15)
#   REPS=1 EPOCHS=3 bash pruebas/run_all.sh # prueba rapida
set -euo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"

bash "$AQUI/inciso0_entorno.sh"
bash "$AQUI/inciso1_analisis_secuencial.sh"
bash "$AQUI/inciso2_secuencial.sh"
bash "$AQUI/inciso3_cuda.sh"
bash "$AQUI/inciso4_comparacion.sh"

echo; echo ">> Listo. Revisa pruebas/resultados/ (tablas.md, figuras/, crudo/, monitor/, logs/)."
