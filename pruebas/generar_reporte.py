#!/usr/bin/env python3
"""Genera las tablas y graficos del informe a partir de los resultados crudos.

Lee   pruebas/resultados/crudo/{secuencial,cuda}_rep*.json   (uno por repeticion)
      pruebas/resultados/monitor/{secuencial,cuda}_rep*.csv   (CPU por nucleo / GPU)
Escribe pruebas/resultados/tablas.md, tiempos_por_fase.csv y figuras/*.png

Tambien se importa desde el notebook asl_cuda.ipynb (seccion de comparacion).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent / "resultados"

# (etiqueta de la fila, filtro, nivel)  - filtro: fase=... / grupo=... / "total" / "total_sin_prep"
FILAS = [
    ("Preparación de datos", {"grupo": "prep_datos"}, 0),
    ("· carga + decodificación + resize", {"fase": "carga_resize"}, 1),
    ("· partición estratificada", {"fase": "split_estratificado"}, 1),
    ("· guardado .npz", {"fase": "guardar_npz"}, 1),
    ("· lectura .npz", {"fase": "leer_npz"}, 1),
    ("Transferencia CPU→GPU (H2D)", {"grupo": "h2d"}, 0),
    ("Entrenamiento", {"grupo": "entrenamiento"}, 0),
    ("Evaluación en test", {"grupo": "evaluacion"}, 0),
    ("TOTAL del pipeline", "total", 0),
    ("TOTAL sin preparación de datos", "total_sin_prep", 0),
]
COLORES_GRUPO = {"prep_datos": "#4C78A8", "h2d": "#F58518", "entrenamiento": "#54A24B", "evaluacion": "#B279A2",
                 "jit": "#9D9D9D"}
NOMBRE_GRUPO = {"prep_datos": "Preparación (CPU)", "h2d": "Transferencia H2D", "entrenamiento": "Entrenamiento",
                "evaluacion": "Evaluación", "jit": "Compilación JIT"}


def cargar_runs(etiqueta, res=RES):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((Path(res) / "crudo").glob(f"{etiqueta}_rep*.json"))]


def _valor(run, filtro):
    f = run["fases"]
    if filtro == "total":
        return sum(x["duracion_s"] for x in f if x["en_total"])
    if filtro == "total_sin_prep":
        return sum(x["duracion_s"] for x in f if x["en_total"] and x["grupo"] != "prep_datos")
    clave, valor = next(iter(filtro.items()))
    sel = [x["duracion_s"] for x in f if x[clave] == valor]
    return sum(sel) if sel else None


def _media_std(runs, filtro):
    v = [x for x in (_valor(r, filtro) for r in runs) if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)


def tabla_tiempos(seq, cu):
    """DataFrame con tiempo medio +- desviacion de cada fase y el speedup (t_secuencial / t_cuda)."""
    filas = []
    for nombre, filtro, nivel in FILAS:
        ms, ss, ns = _media_std(seq, filtro)
        mc, sc, nc = _media_std(cu, filtro)
        sp = ms / mc if (ms and mc) else None
        filas.append({"fase": nombre, "nivel": nivel, "seq_media_s": ms, "seq_std_s": ss, "seq_n": ns,
                      "cuda_media_s": mc, "cuda_std_s": sc, "cuda_n": nc, "speedup": sp})
    return pd.DataFrame(filas)


def _fmt(m, s):
    return "—" if m is None else (f"{m:.2f} ± {s:.2f}" if s is not None else f"{m:.2f}")


def md_tiempos(df):
    L = ["| Fase | Secuencial (s) | CUDA (s) | Speedup |", "|---|---:|---:|---:|"]
    for _, r in df.iterrows():
        sp = "—" if pd.isna(r.speedup) else f"{r.speedup:.2f}×"
        nombre = f"**{r.fase}**" if r.nivel == 0 and r.fase.startswith("TOTAL") else r.fase
        L.append(f"| {nombre} | {_fmt(None if pd.isna(r.seq_media_s) else r.seq_media_s, r.seq_std_s)} | "
                 f"{_fmt(None if pd.isna(r.cuda_media_s) else r.cuda_media_s, r.cuda_std_s)} | {sp} |")
    return "\n".join(L)


def tabla_epocas(seq, cu):
    def ep(runs):
        v = [np.mean(r["historial"]["epoch_time"]) for r in runs if r.get("historial")]
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) if v else (None, None)
    (ms, ss), (mc, sc) = ep(seq), ep(cu)
    return ms, ss, mc, sc


def md_calidad(seq, cu):
    def g(runs, k):
        v = [r[k] for r in runs if k in r]
        return f"{np.mean(v) * 100:.2f}%" if v else "—"

    def gv(runs):
        v = [r["historial"]["val_acc"][-1] for r in runs if r.get("historial")]
        return f"{np.mean(v) * 100:.2f}%" if v else "—"
    L = ["| Métrica | Secuencial (PyTorch CPU) | CUDA (kernels propios) |", "|---|---:|---:|",
         f"| Accuracy de validación (última época) | {gv(seq)} | {gv(cu)} |",
         f"| Accuracy en test | {g(seq, 'test_acc')} | {g(cu, 'test_acc')} |"]
    return "\n".join(L)


def md_hardware(seq, cu):
    L = []
    for nombre, runs in (("Secuencial", seq), ("CUDA", cu)):
        if not runs:
            continue
        m, c = runs[0]["meta"], runs[0]["config"]
        s = m.get("sistema", {})
        L.append(f"- **{nombre}**: {s.get('sistema', '?')} · CPU {s.get('procesador', '?')} · "
                 f"{s.get('nucleos_fisicos', '?')} núcleos físicos / {s.get('nucleos_logicos', '?')} lógicos · "
                 f"RAM {s.get('ram_total_gb', '?')} GB · procesos de preparación: {c['workers']}"
                 + (f" · hilos PyTorch: {c['seq_threads']}" if nombre == "Secuencial" else ""))
        if m.get("gpu"):
            g = m["gpu"]
            L.append(f"  - GPU: {g.get('nombre')} · cc {g.get('compute_capability')} · {g.get('multiprocesadores')} SM · "
                     f"{g.get('memoria_total_mb')} MB")
    return "\n".join(L)


# ----------------------------------------------------------------------------- recursos
def _leer_monitor(etiqueta, rep, res=RES):
    p = Path(res) / "monitor" / f"{etiqueta}_rep{rep}.csv"
    j = Path(res) / "crudo" / f"{etiqueta}_rep{rep}.json"
    if not (p.exists() and j.exists()):
        return None, None
    df = pd.read_csv(p)
    return (df, json.loads(j.read_text(encoding="utf-8"))) if len(df) else (None, None)


def tabla_recursos(etiqueta, rep=1, res=RES):
    """Uso medio/maximo de CPU (total y nucleos activos) y de GPU durante cada fase (evidencia numerica)."""
    df, run = _leer_monitor(etiqueta, rep, res)
    if df is None:
        return None
    nucleos = [c for c in df.columns if c.startswith("cpu_") and c != "cpu_total"]
    filas = []
    for g in dict.fromkeys(f["grupo"] for f in run["fases"] if f["en_total"]):
        spans = [(f["inicio_epoch"], f["fin_epoch"]) for f in run["fases"] if f["grupo"] == g]
        m = np.zeros(len(df), bool)
        for a, b in spans:
            m |= (df.t_epoch >= a) & (df.t_epoch <= b)
        sub = df[m]
        if not len(sub):
            continue
        medio_nuc = sub[nucleos].mean()
        filas.append({"fase": NOMBRE_GRUPO.get(g, g), "muestras": len(sub),
                      "cpu_total_medio": sub.cpu_total.mean(), "cpu_total_max": sub.cpu_total.max(),
                      "nucleos_>50%": int((medio_nuc > 50).sum()), "nucleos_totales": len(nucleos),
                      "gpu_util_media": sub.gpu_util.mean(), "gpu_util_max": sub.gpu_util.max(),
                      "gpu_mem_max_mb": sub.gpu_mem_mb.max(), "ram_max_mb": sub.ram_usada_mb.max()})
    return pd.DataFrame(filas)


def md_recursos(df):
    if df is None or df.empty:
        return "_(sin datos del monitor)_"
    L = ["| Fase | CPU medio (%) | CPU máx (%) | Núcleos con >50% medio | GPU util. media (%) | GPU util. máx (%) | VRAM máx (MB) |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        nz = lambda x: "—" if pd.isna(x) else f"{x:.0f}"  # noqa: E731
        L.append(f"| {r.fase} | {r.cpu_total_medio:.0f} | {r.cpu_total_max:.0f} | {r['nucleos_>50%']}/{r.nucleos_totales} | "
                 f"{nz(r.gpu_util_media)} | {nz(r.gpu_util_max)} | {nz(r.gpu_mem_max_mb)} |")
    return "\n".join(L)


# ----------------------------------------------------------------------------- graficos
def _plt():
    import matplotlib.pyplot as plt
    return plt


def fig_tiempos(df):
    plt = _plt()
    d = df[(df.nivel == 0)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(d))
    w = 0.38
    ax.bar(x - w / 2, d.seq_media_s.fillna(0), w, yerr=d.seq_std_s.fillna(0), label="Secuencial (CPU)", color="#4C78A8")
    ax.bar(x + w / 2, d.cuda_media_s.fillna(0), w, yerr=d.cuda_std_s.fillna(0), label="CUDA", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(" (H2D)", "").replace(" del pipeline", "").replace("TOTAL sin preparación de datos", "TOTAL sin prep.")
                        for f in d.fase], rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("segundos (escala log)")
    ax.set_title("Tiempo por fase: secuencial vs CUDA")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_speedup(df):
    plt = _plt()
    d = df[(df.nivel == 0) & df.speedup.notna()]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    b = ax.bar(range(len(d)), d.speedup, color="#F58518")
    ax.axhline(1, color="k", lw=1, ls="--")
    for r, v in zip(b, d.speedup):
        ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.1f}×", ha="center", va="bottom")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([f.replace(" del pipeline", "").replace("TOTAL sin preparación de datos", "TOTAL sin prep.") for f in d.fase],
                       rotation=20, ha="right")
    ax.set_ylabel("speedup = t_secuencial / t_CUDA")
    ax.set_title("Aceleración por fase (línea punteada = sin mejora)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_epocas(seq, cu):
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for runs, nombre, col in ((seq, "Secuencial", "#4C78A8"), (cu, "CUDA", "#54A24B")):
        hs = [r["historial"] for r in runs if r.get("historial")]
        if not hs:
            continue
        for k, ax, tit in (("epoch_time", axs[0], "Tiempo por época (s)"), ("val_acc", axs[1], "Accuracy de validación")):
            M = np.array([h[k] for h in hs])
            ax.plot(range(1, M.shape[1] + 1), M.mean(0), marker="o", ms=3, label=nombre, color=col)
            ax.set_title(tit)
            ax.set_xlabel("época")
    axs[0].set_yscale("log")
    for a in axs:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    return fig


def fig_recursos(etiqueta, rep=1, res=RES):
    df, run = _leer_monitor(etiqueta, rep, res)
    if df is None:
        return None
    plt = _plt()
    nucleos = [c for c in df.columns if c.startswith("cpu_") and c != "cpu_total"]
    t0 = run["fases"][0]["inicio_epoch"] - 2
    t = df.t_epoch - t0
    fig, ax = plt.subplots(3, 1, figsize=(11.5, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]})
    im = ax[0].imshow(df[nucleos].T.values, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=100,
                      extent=[t.iloc[0], t.iloc[-1], -0.5, len(nucleos) - 0.5])
    ax[0].set_ylabel("núcleo lógico")
    fig.colorbar(im, ax=ax[0], label="uso (%)", pad=0.01)
    ax[1].plot(t, df.cpu_total, color="#4C78A8", label="CPU total (%)")
    if df.gpu_util.notna().any():
        ax[1].plot(t, df.gpu_util, color="#54A24B", label="GPU (%)")
    ax[1].set_ylabel("uso (%)")
    ax[1].set_ylim(0, 105)
    ax[1].legend(loc="upper right")
    ax[2].plot(t, df.ram_usada_mb / 1024, color="#B279A2", label="RAM usada (GB)")
    ax[2].set_ylabel("RAM (GB)")
    if df.gpu_mem_mb.notna().any():
        a2 = ax[2].twinx()
        a2.plot(t, df.gpu_mem_mb / 1024, color="#F58518", label="VRAM usada (GB)")
        a2.set_ylabel("VRAM (GB)")
        a2.legend(loc="upper right")
    ax[2].legend(loc="upper left")
    ax[2].set_xlabel("segundos desde el inicio de la corrida")
    visto = set()
    for f in run["fases"]:
        if f["grupo"] == "jit" or not f["en_total"]:
            c = "#9D9D9D"
        else:
            c = COLORES_GRUPO.get(f["grupo"], "#999")
        for a in ax:
            a.axvspan(f["inicio_epoch"] - t0, f["fin_epoch"] - t0, color=c, alpha=0.12)
        if f["grupo"] not in visto:
            visto.add(f["grupo"])
            ax[0].text(f["inicio_epoch"] - t0, len(nucleos) + 0.1, NOMBRE_GRUPO.get(f["grupo"], f["grupo"]),
                       fontsize=8, rotation=0, va="bottom", color=c)
    ax[0].set_title(f"Uso de recursos durante la corrida «{etiqueta}» (rep. {rep}) — fases sombreadas", pad=18)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- main
def main():
    import matplotlib
    matplotlib.use("Agg")
    seq, cu = cargar_runs("secuencial"), cargar_runs("cuda")
    if not seq or not cu:
        print(f"[reporte] faltan resultados: secuencial={len(seq)} corridas, cuda={len(cu)} corridas.\n"
              "          Ejecuta primero inciso2_secuencial.sh e inciso3_cuda.sh.", file=sys.stderr)
        sys.exit(1)
    (RES / "figuras").mkdir(parents=True, exist_ok=True)
    df = tabla_tiempos(seq, cu)
    df.to_csv(RES / "tiempos_por_fase.csv", index=False)
    ms, ss, mc, sc = tabla_epocas(seq, cu)
    jit = [r_["duracion_s"] for r in cu for r_ in r["fases"] if r_["grupo"] == "jit"]
    L = ["# Resultados: secuencial vs CUDA\n",
         f"Repeticiones: secuencial = {len(seq)}, CUDA = {len(cu)} (media ± desviación estándar muestral).\n",
         "## Tabla 1. Comparativa de tiempos y speedup por fase\n", md_tiempos(df), ""]
    if jit:
        L.append(f"_La compilación JIT de los kernels (una sola vez por proceso, fuera de los totales) tomó {np.mean(jit):.2f} s._\n")
    if ms and mc:
        L.append(f"## Tabla 2. Tiempo por época de entrenamiento\n\n| | Secuencial | CUDA | Speedup |\n|---|---:|---:|---:|\n"
                 f"| s/época (media ± desv.) | {ms:.2f} ± {ss:.2f} | {mc:.3f} ± {sc:.3f} | {ms / mc:.1f}× |\n")
    L += ["## Tabla 3. Calidad del modelo (misma arquitectura, mismos datos)\n", md_calidad(seq, cu), "",
          "## Tabla 4. Hardware y condiciones\n", md_hardware(seq, cu), ""]
    for etiqueta, titulo in (("secuencial", "Tabla 5. Uso de recursos — versión secuencial"), ("cuda", "Tabla 6. Uso de recursos — versión CUDA")):
        L += [f"## {titulo}\n", md_recursos(tabla_recursos(etiqueta)), ""]
    (RES / "tablas.md").write_text("\n".join(L), encoding="utf-8")

    for nombre, fig in (("tiempos", fig_tiempos(df)), ("speedup", fig_speedup(df)), ("epocas", fig_epocas(seq, cu)),
                        ("recursos_secuencial", fig_recursos("secuencial")), ("recursos_cuda", fig_recursos("cuda"))):
        if fig is not None:
            fig.savefig(RES / "figuras" / f"{nombre}.png", dpi=150)
            print(f"[reporte] figura: figuras/{nombre}.png")
    print("\n".join(L))
    print(f"\n[reporte] tablas en {RES / 'tablas.md'}")


if __name__ == "__main__":
    main()
