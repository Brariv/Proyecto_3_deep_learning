#!/usr/bin/env python3
"""Lee una vez las imagenes de la submuestra para que el cache del sistema operativo este
"caliente" antes de medir. Sin esto, la primera medicion pagaria el disco frio y la segunda no,
y el speedup de la fase de preparacion saldria inflado o deformado segun el orden de ejecucion.
No se mide ni se guarda nada."""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import data_utils as du  # noqa: E402
from bench_utils import cargar_config  # noqa: E402


def main():
    c = cargar_config("cuda")
    t0 = time.time()
    X, _, _ = du.build_subsample_parallel(c.n_per_class, c.img_size, c.seed, Path(c.raw_dir), n_workers=c.workers)
    print(f"[cache] {len(X)} imagenes leidas en {time.time() - t0:.1f}s (cache del SO caliente)")


if __name__ == "__main__":
    main()
