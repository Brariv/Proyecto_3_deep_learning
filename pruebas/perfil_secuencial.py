#!/usr/bin/env python3
"""Inciso 1 - Analisis de la solucion secuencial: donde se va el tiempo y cuantas veces se repite cada operacion.

Produce (en pruebas/resultados/):
  perfil_secuencial.md   tablas "Operaciones identificadas", "Operaciones repetitivas" y tiempos medidos
  perfil_secuencial.csv  tiempo medido por etapa
Todo corre en CPU con los mismos parametros del benchmark (variables P3_*).
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import data_utils as du  # noqa: E402
import models as mdl  # noqa: E402
import train_utils as tu  # noqa: E402
from bench_utils import cargar_config  # noqa: E402


def medir_prep(cfg, n_img=300):
    """Costo por imagen de leer + decodificar + redimensionar (muestra aleatoria)."""
    archivos = du.list_class_files(Path(cfg.raw_dir))
    todos = [f for c in du.CLASSES for f in archivos[c]]
    rng = np.random.default_rng(cfg.seed)
    muestra = [todos[i] for i in rng.choice(len(todos), size=min(n_img, len(todos)), replace=False)]
    t_leer = t_dec = t_res = 0.0
    for f in muestra:
        t0 = time.perf_counter(); datos = f.read_bytes(); t1 = time.perf_counter()
        import io
        img = Image.open(io.BytesIO(datos)); img = img.convert("RGB"); t2 = time.perf_counter()
        img.resize((cfg.img_size, cfg.img_size), Image.BILINEAR); t3 = time.perf_counter()
        t_leer += t1 - t0; t_dec += t2 - t1; t_res += t3 - t2
    n = len(muestra)
    return {"leer_archivo": t_leer / n, "decodificar_jpg_a_rgb": t_dec / n, "resize_bilineal": t_res / n}, len(todos)


def medir_entrenamiento(cfg, iters=30):
    """Desglose de un paso de entrenamiento de la SmallCNN en CPU (batch real del benchmark)."""
    torch.set_num_threads(cfg.seq_threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n = cfg.batch_size * 4
    X = rng.integers(0, 256, size=(n, cfg.img_size, cfg.img_size, 3), dtype=np.uint8)
    y = rng.integers(0, len(du.CLASSES), size=n).astype(np.int64)
    ds = tu.ASLTensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    model = mdl.SmallCNN(len(du.CLASSES), cfg.img_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = torch.nn.CrossEntropyLoss()
    model.train()
    ac = {"armar_lote (Dataset+DataLoader: /255, permutar a CHW, apilar)": 0.0, "forward (conv+ReLU+pool+lineal)": 0.0,
          "perdida (entropia cruzada)": 0.0, "backward (autograd)": 0.0, "optimizador Adam": 0.0}
    cuenta = 0
    it = iter(loader)
    for i in range(iters + 3):
        t0 = time.perf_counter()
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader); xb, yb = next(it)
        t1 = time.perf_counter()
        opt.zero_grad(); out = model(xb); t2 = time.perf_counter()
        loss = crit(out, yb); t3 = time.perf_counter()
        loss.backward(); t4 = time.perf_counter()
        opt.step(); t5 = time.perf_counter()
        if i >= 3:  # se descartan 3 iteraciones de calentamiento
            for k, d in zip(ac, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                ac[k] += d
            cuenta += 1
    return {k: v / cuenta for k, v in ac.items()}, model, loader


def perfil_operadores(model, loader, cfg, iters=8):
    """torch.profiler: tiempo propio de CPU por operador (conv, relu, pool, addmm, ...)."""
    from torch.profiler import ProfilerActivity, profile
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    it = iter(loader)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(iters):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(loader); xb, yb = next(it)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    filas = []
    total = sum(e.self_cpu_time_total for e in prof.key_averages()) or 1
    for e in sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)[:8]:
        filas.append((e.key, e.count, e.self_cpu_time_total / 1000.0, 100.0 * e.self_cpu_time_total / total))
    return filas


def conteos(cfg, n_archivos):
    n_img = cfg.n_per_class * len(du.CLASSES)
    por_clase = cfg.n_per_class
    tr = int(por_clase * 0.7) * len(du.CLASSES)
    va = int(por_clase * 0.15) * len(du.CLASSES)
    te = n_img - tr - va
    bt, bv = math.ceil(tr / cfg.batch_size), math.ceil(va / cfg.batch_size)
    s, c1, c2 = cfg.img_size, 16, 32
    mac = {"conv1": s * s * 3 * c1 * 9, "conv2": (s // 2) ** 2 * c1 * c2 * 9,
           "fc1": c2 * (s // 4) ** 2 * 128, "fc2": 128 * len(du.CLASSES)}
    return dict(n_img=n_img, tr=tr, va=va, te=te, bt=bt, bv=bv, mac=mac, mac_img=sum(mac.values()), n_archivos=n_archivos)


def main():
    cfg = cargar_config("secuencial")
    print(f"[perfil] SmallCNN en CPU | hilos={cfg.seq_threads} | batch={cfg.batch_size} | epocas={cfg.epochs}")
    prep, n_archivos = medir_prep(cfg)
    paso, model, loader = medir_entrenamiento(cfg)
    ops = perfil_operadores(model, loader, cfg)
    c = conteos(cfg, n_archivos)
    ep = cfg.epochs

    L = []
    L.append("# Analisis de la solucion secuencial (CPU)\n")
    L.append(f"Parametros: {c['n_img']:,} imagenes ({cfg.n_per_class}/clase, {len(du.CLASSES)} clases), {cfg.img_size}x{cfg.img_size}, "
             f"batch {cfg.batch_size}, {ep} epocas, SmallCNN ({sum(p.numel() for p in model.parameters()):,} parametros), "
             f"{cfg.seq_threads} hilo(s) de PyTorch.\n")
    L.append("## Tabla 1. Operaciones identificadas (de la imagen al resultado)\n")
    L.append("| No. | Operacion identificada | Descripcion de su funcionamiento |\n|---|---|---|")
    S, F = cfg.img_size, 32 * (cfg.img_size // 4) ** 2
    pasos = [
        ("Listado y submuestreo", "Se listan los .jpg de cada clase y se eligen al azar `n_per_class` por clase (rng con semilla)."),
        ("Lectura + decodificacion JPG", "Se lee el archivo y se decodifica a pixeles RGB 200x200 (CPU, acceso a disco)."),
        ("Redimensionado", f"Interpolacion bilineal 200x200 -> {S}x{S} por imagen."),
        ("Particion estratificada", "Indices train/val/test 70/15/15 por clase; se guardan en .npz."),
        ("Armado de lote", f"`Dataset.__getitem__`: uint8 HWC -> float32 CHW /255; `DataLoader` mezcla y apila {cfg.batch_size} imagenes."),
        ("Forward de la CNN", f"Conv3x3(3->16)+ReLU+MaxPool, Conv3x3(16->32)+ReLU+MaxPool, Linear({F}->128)+ReLU+Dropout, Linear(128->29)."),
        ("Perdida", "Entropia cruzada (softmax + log-verosimilitud) promediada sobre el lote."),
        ("Backward", "Retropropagacion: gradientes de convoluciones, capas lineales, ReLU, pooling y dropout."),
        ("Actualizacion Adam", "Momentos de primer/segundo orden y actualizacion de los 8 tensores de parametros."),
        ("Evaluacion", "Forward sin dropout sobre validacion (cada epoca) y test; argmax y accuracy."),
    ]
    for i, (a, b) in enumerate(pasos, 1):
        L.append(f"| {i} | {a} | {b} |")

    L.append("\n## Tabla 2. Operaciones repetitivas y cantidad de repeticiones\n")
    L.append("| Operacion repetitiva | Datos que procesa | Cantidad de repeticiones |\n|---|---|---|")
    L.append(f"| Lectura + decodificacion + resize | 1 imagen JPG 200x200 | {c['n_img']:,} (una vez por imagen de la submuestra; el dataset completo tiene {c['n_archivos']:,}) |")
    L.append(f"| Armado de lote (/255, CHW) | 1 imagen {cfg.img_size}x{cfg.img_size}x3 | {(c['tr'] + c['va']) * ep + c['te']:,} = ({c['tr']:,} train + {c['va']:,} val) x {ep} epocas + {c['te']:,} test |")
    L.append(f"| Convolucion 3x3 + ReLU + MaxPool (x2 capas) | lote [{cfg.batch_size},3,{cfg.img_size},{cfg.img_size}] | forward: {c['bt'] * ep:,} lotes de train + {c['bv'] * ep:,} de val + {c['bv']} de test; backward: {c['bt'] * ep:,} |")
    L.append(f"| Capas lineales (matmul) | [{cfg.batch_size},{F}] x [{F},128] y [{cfg.batch_size},128] x [128,29] | igual que arriba (forward) y 2 matmul extra por capa en backward |")
    L.append(f"| Producto-acumulacion (MAC) por imagen | {c['mac_img']:,} MAC forward (conv1 {c['mac']['conv1']:,}; conv2 {c['mac']['conv2']:,}; fc1 {c['mac']['fc1']:,}; fc2 {c['mac']['fc2']:,}) | {c['tr'] * ep:,} imagenes de train x ~3 (fwd+bwd) = {c['mac_img'] * c['tr'] * ep * 3:.3e} MAC |")
    L.append(f"| Actualizacion Adam | 8 tensores, {sum(p.numel() for p in model.parameters()):,} parametros | {c['bt'] * ep:,} pasos |")

    L.append("\n## Tabla 3. Tiempo medido por imagen en la preparacion de datos (CPU, 1 proceso)\n")
    L.append("| Etapa | ms por imagen | % |\n|---|---|---|")
    tp = sum(prep.values())
    for k, v in prep.items():
        L.append(f"| {k} | {v * 1000:.3f} | {100 * v / tp:.1f}% |")
    L.append(f"| **Total por imagen** | **{tp * 1000:.3f}** | 100% |")
    L.append(f"\nProyeccion para {c['n_img']:,} imagenes en un solo proceso: **{tp * c['n_img']:.1f} s** (sin contar particion ni guardado).\n")

    L.append("## Tabla 4. Desglose medido de un paso de entrenamiento (CPU)\n")
    L.append("| Etapa | ms por lote | % |\n|---|---|---|")
    tt = sum(paso.values())
    for k, v in paso.items():
        L.append(f"| {k} | {v * 1000:.2f} | {100 * v / tt:.1f}% |")
    L.append(f"| **Total por lote** | **{tt * 1000:.2f}** | 100% |")
    L.append(f"\nProyeccion de entrenamiento: {tt * c['bt'] * ep:.1f} s ({c['bt'] * ep:,} pasos), sin contar la validacion de cada epoca.\n")

    L.append("## Tabla 5. Operadores que mas tiempo de CPU consumen (torch.profiler)\n")
    L.append("| Operador | Llamadas | CPU propio (ms) | % |\n|---|---|---|---|")
    for k, n, ms, pc in ops:
        L.append(f"| `{k}` | {n} | {ms:.1f} | {pc:.1f}% |")

    destino = REPO / "pruebas" / "resultados"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "perfil_secuencial.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(destino / "perfil_secuencial.csv", "w", encoding="utf-8") as f:
        f.write("grupo,etapa,segundos\n")
        for k, v in prep.items():
            f.write(f"prep_por_imagen,{k},{v:.6f}\n")
        for k, v in paso.items():
            f.write(f"entrenamiento_por_lote,{k},{v:.6f}\n")
    print("\n".join(L))
    print(f"\n[perfil] guardado en {destino / 'perfil_secuencial.md'}")


if __name__ == "__main__":
    main()
