#!/usr/bin/env python3
"""Reporta el hardware y el software que reconoce el entorno (evidencia del inciso 0).

Imprime un resumen legible y escribe pruebas/resultados/entorno.json con, entre otros,
"cuda_disponible" (lo consulta inciso3_cuda.sh antes de correr la version GPU)."""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from bench_utils import info_sistema  # noqa: E402


def main():
    info = info_sistema()
    print("=== CPU / sistema ===")
    for k, v in info.items():
        print(f"  {k}: {v}")

    try:
        import torch
        info["torch"] = {"version": torch.__version__, "hilos_cpu_por_defecto": torch.get_num_threads(),
                         "cuda_en_torch": torch.cuda.is_available()}
        print(f"=== PyTorch (linea base secuencial) ===\n  {info['torch']}")
    except ImportError:
        print("=== PyTorch === NO instalado (necesario para el benchmark secuencial)")

    cuda_ok = False
    try:
        import numba
        from numba import cuda
        cuda_ok = bool(cuda.is_available())
        info["numba"] = {"version": numba.__version__, "cuda_disponible": cuda_ok}
        print(f"=== Numba CUDA ===\n  numba {numba.__version__} | cuda.is_available() = {cuda_ok}")
        if cuda_ok:
            dev = cuda.get_current_device()
            try:
                libre, total = cuda.current_context().get_memory_info()
            except Exception:  # noqa: BLE001
                libre, total = 0, 0
            info["gpu_numba"] = {
                "nombre": (lambda n: n.decode() if isinstance(n, bytes) else str(n))(getattr(dev, "name", "desconocida")),
                "compute_capability": list(getattr(dev, "compute_capability", ())),
                "multiprocesadores": getattr(dev, "MULTIPROCESSOR_COUNT", None),
                "max_hilos_por_bloque": getattr(dev, "MAX_THREADS_PER_BLOCK", None),
                "memoria_total_mb": round(total / 2 ** 20) if 0 < total < float("inf") else None,
                "memoria_libre_mb": round(libre / 2 ** 20) if 0 < libre < float("inf") else None}
            print("  GPU:", info["gpu_numba"])
    except Exception as e:  # noqa: BLE001  (queremos el motivo, sea cual sea)
        print(f"=== Numba CUDA === no disponible: {type(e).__name__}: {e}")
    info["cuda_disponible"] = cuda_ok

    print("=== nvidia-smi ===")
    try:
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15).stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        print("  nvidia-smi no encontrado (sin driver NVIDIA en este equipo)")

    destino = REPO / "pruebas" / "resultados"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "entorno.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[entorno] guardado en {destino / 'entorno.json'} | {platform.platform()} | PID {os.getpid()}")


if __name__ == "__main__":
    main()
