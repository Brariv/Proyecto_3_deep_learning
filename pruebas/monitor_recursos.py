#!/usr/bin/env python3
"""Monitor de recursos (evidencia de ejecucion): CPU por nucleo, RAM y GPU NVIDIA.

Muestrea cada --intervalo segundos y escribe un CSV. Se detiene cuando aparece el archivo
--parar (funciona igual en macOS, Linux, Git Bash y WSL; no depende de senales).
Equivalente programatico al Administrador de Tareas / `nvidia-smi -l 1`.
"""
import argparse
import csv
import subprocess
import time
from pathlib import Path

import psutil

CAMPOS_GPU = "utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu"


def leer_gpu():
    """[util_gpu %, util_mem %, mem_usada MB, mem_total MB, potencia W, temp C] o None."""
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={CAMPOS_GPU}", "--format=csv,noheader,nounits", "-i", "0"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        vals = []
        for x in r.stdout.strip().splitlines()[0].split(","):
            x = x.strip()
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(float("nan"))
        return vals
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", required=True)
    ap.add_argument("--intervalo", type=float, default=0.5)
    ap.add_argument("--parar", required=True, help="al existir este archivo, el monitor termina")
    ap.add_argument("--max-seg", type=float, default=6 * 3600)
    a = ap.parse_args()

    n = psutil.cpu_count(logical=True)
    cab = ["t_epoch", "t_rel", "cpu_total"] + [f"cpu_{i}" for i in range(n)] + [
        "ram_usada_mb", "gpu_util", "gpu_mem_util", "gpu_mem_mb", "gpu_mem_total_mb", "gpu_potencia_w", "gpu_temp_c"]
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    nan = float("nan")
    psutil.cpu_percent(percpu=True)            # primera lectura descartada (psutil la inicializa)
    t0 = time.time()
    with open(a.salida, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cab)
        while not Path(a.parar).exists() and time.time() - t0 < a.max_seg:
            time.sleep(a.intervalo)
            por_nucleo = psutil.cpu_percent(percpu=True)
            gpu = leer_gpu() or [nan] * 6
            w.writerow([f"{time.time():.3f}", f"{time.time() - t0:.3f}", f"{sum(por_nucleo) / len(por_nucleo):.1f}"]
                       + [f"{x:.1f}" for x in por_nucleo]
                       + [f"{psutil.virtual_memory().used / 2 ** 20:.0f}"] + [f"{x:.1f}" for x in gpu])
            f.flush()


if __name__ == "__main__":
    main()
