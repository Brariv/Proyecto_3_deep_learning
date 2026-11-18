#!/usr/bin/env bash
# Inciso 4: comparacion CUDA vs secuencial. Calcula tiempos medios, speedup por fase,
# uso de recursos por fase y genera tablas (resultados/tablas.md) y graficos (resultados/figuras/).
# Requiere haber corrido inciso2_secuencial.sh e inciso3_cuda.sh en la MISMA maquina.
source "$(dirname "$0")/comun.sh"

banner "INCISO 4: comparacion secuencial vs CUDA"
"$PY" "$(ruta "$PRUEBAS_DIR/generar_reporte.py")" 2>&1 | tee "$LOGS/comparacion.log"
echo; echo "Tablas : $RESULTADOS/tablas.md"
echo "Figuras: $RESULTADOS/figuras/"
