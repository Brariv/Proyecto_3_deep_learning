"""Utilidades de medicion compartidas por el notebook secuencial (CPU) y el notebook CUDA.

* ConfigBench / cargar_config : parametros del experimento, leidos de variables P3_* que
  fijan los scripts bash (asi ambas versiones corren con LOS MISMOS datos y condiciones).
* RegistroFases               : cronometra cada fase con marcas de tiempo absolutas para poder
  alinearlas con el monitor de recursos (CPU por nucleo / GPU).
* info_sistema                : hardware y versiones, para el informe.
"""
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTADOS = REPO_ROOT / "pruebas" / "resultados"


def _env(nombre, defecto, tipo=str):
    v = os.getenv(nombre)
    return defecto if v in (None, "") else tipo(v)


@dataclass
class ConfigBench:
    etiqueta: str          # "secuencial" | "cuda"
    arquitectura: str
    n_per_class: int
    img_size: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    workers: int           # procesos CPU para la preparacion de imagenes
    seq_threads: int       # hilos de PyTorch en la linea base secuencial
    rep: int
    raw_dir: str
    processed_dir: str
    salida_dir: str


def cargar_config(etiqueta: str) -> ConfigBench:
    """Lee P3_* del entorno. Sin variables definidas usa los valores del Lab 3 (600 img/clase, 64 px, ...)."""
    cpu = os.cpu_count() or 1
    workers = _env("P3_WORKERS", 0, int) or cpu
    return ConfigBench(
        etiqueta=etiqueta,
        arquitectura="SmallCNN",
        n_per_class=_env("P3_N_PER_CLASS", 600, int),
        img_size=_env("P3_IMG_SIZE", 64, int),
        epochs=_env("P3_EPOCHS", 15, int),
        batch_size=_env("P3_BATCH", 128, int),
        lr=_env("P3_LR", 1e-3, float),
        weight_decay=_env("P3_WD", 1e-4, float),
        seed=_env("P3_SEED", 42, int),
        workers=1 if etiqueta == "secuencial" else min(workers, 61),
        seq_threads=_env("P3_SEQ_THREADS", 1, int),
        rep=_env("P3_REP", 1, int),
        raw_dir=_env("P3_RAW_DIR", str(REPO_ROOT / "data" / "raw" / "asl_alphabet_train" / "asl_alphabet_train")),
        processed_dir=str(REPO_ROOT / "data" / "processed" / f"bench_{etiqueta}"),
        salida_dir=_env("P3_SALIDA_DIR", str(RESULTADOS)),
    )


def _nvidia_smi(campos):
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={campos}", "--format=csv,noheader,nounits", "-i", "0"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def info_sistema() -> dict:
    info = {
        "python": sys.version.split()[0],
        "sistema": f"{platform.system()} {platform.release()}",
        "procesador": platform.processor() or platform.machine(),
        "nucleos_logicos": os.cpu_count(),
    }
    try:
        import psutil
        info["nucleos_fisicos"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        pass
    smi = _nvidia_smi("name,driver_version,memory.total")
    if smi:
        info["gpu_nvidia_smi"] = {"nombre": smi[0], "driver": smi[1], "memoria_total_mb": smi[2]}
    return info


class RegistroFases:
    """Cronometra fases. Cada fase queda con inicio/fin en hora absoluta (time.time())."""

    def __init__(self, config: ConfigBench, meta: dict | None = None):
        self.config = config
        self.meta = {"sistema": info_sistema(), **(meta or {})}
        self.fases = []

    @contextmanager
    def fase(self, nombre: str, grupo: str, en_total: bool = True, sync=None):
        """sync: funcion a llamar antes y despues (p. ej. cuda.synchronize) para medir la GPU de verdad."""
        if sync:
            sync()
        ini, p0 = time.time(), time.perf_counter()
        print(f"[FASE] >>> {nombre} ({grupo})", flush=True)
        try:
            yield
        finally:
            if sync:
                sync()
            dur = time.perf_counter() - p0
            self.fases.append({"fase": nombre, "grupo": grupo, "en_total": en_total,
                               "inicio_epoch": ini, "fin_epoch": ini + dur, "duracion_s": dur})
            print(f"[FASE] <<< {nombre}: {dur:.3f} s", flush=True)

    def duracion(self, grupo: str) -> float:
        return sum(f["duracion_s"] for f in self.fases if f["grupo"] == grupo)

    def total(self) -> float:
        return sum(f["duracion_s"] for f in self.fases if f["en_total"])

    def guardar(self, extra: dict | None = None) -> Path:
        c = self.config
        destino = Path(c.salida_dir) / "crudo"
        destino.mkdir(parents=True, exist_ok=True)
        ruta = destino / f"{c.etiqueta}_rep{c.rep}.json"
        datos = {"config": asdict(c), "meta": self.meta, "fases": self.fases,
                 "total_s": self.total(), **(extra or {})}
        ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[BENCH] resultados guardados en {ruta}", flush=True)
        return ruta
