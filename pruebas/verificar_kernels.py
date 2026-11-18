#!/usr/bin/env python3
"""Verifica que los kernels CUDA escritos a mano calculan lo mismo que PyTorch.

PyTorch se usa AQUI solo como "oraculo" en CPU (no participa en el entrenamiento CUDA):
  1. logits y perdida del forward,
  2. gradientes de los 8 tensores de parametros (backward),
  3. un paso de Adam (con weight_decay),
  4. lote parcial (n < batch_max),
  5. estadistica del dropout y que en evaluacion no hace nada,
  6. el conteo de parametros coincide con models.SmallCNN (1,057,533),
  7. unos pasos de entrenamiento reducen la perdida.

Con GPU real usa las dimensiones completas de la SmallCNN; con el simulador de Numba
(NUMBA_ENABLE_CUDASIM=1) usa una red diminuta para que termine rapido.
Salida: codigo 0 si todo coincide, 1 si algo falla.
"""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SIM = os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1"
if SIM:
    # En el simulador cada hilo es un hilo de Python: se reduce el bloque de reduccion para que sea rapido.
    import cuda_kernels
    cuda_kernels.RED = 32

import torch  # noqa: E402  (solo como referencia en CPU)
import torch.nn as nn  # noqa: E402
from numba import cuda  # noqa: E402

from cuda_cnn import CudaSmallCNN, NOMBRES_PARAMS  # noqa: E402

fallos = []


def check(nombre, ok, detalle=""):
    print(f"  [{'OK ' if ok else 'FALLO'}] {nombre} {detalle}")
    if not ok:
        fallos.append(nombre)


def rel_err(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def modelo_referencia(net, p):
    """Red de PyTorch (CPU) con las mismas dimensiones y los mismos pesos que la CUDA."""
    ref = nn.Sequential(
        nn.Conv2d(net.C, net.c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(net.c1, net.c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(net.F, net.hidden), nn.ReLU(), nn.Linear(net.hidden, net.K),
    ).double()
    T = lambda k, shape: torch.from_numpy(p[k].astype(np.float64).reshape(shape))  # noqa: E731
    with torch.no_grad():
        ref[0].weight.copy_(T("w1", (net.c1, net.C, 3, 3))); ref[0].bias.copy_(T("b1", (net.c1,)))
        ref[3].weight.copy_(T("w2", (net.c2, net.c1, 3, 3))); ref[3].bias.copy_(T("b2", (net.c2,)))
        ref[7].weight.copy_(T("wf1", (net.hidden, net.F))); ref[7].bias.copy_(T("bf1", (net.hidden,)))
        ref[9].weight.copy_(T("wf2", (net.K, net.hidden))); ref[9].bias.copy_(T("bf2", (net.K,)))
    return ref


def grads_referencia(ref):
    g = {}
    for k, (cap, attr) in zip(NOMBRES_PARAMS, [(0, "weight"), (0, "bias"), (3, "weight"), (3, "bias"),
                                               (7, "weight"), (7, "bias"), (9, "weight"), (9, "bias")]):
        g[k] = getattr(ref[cap], attr).grad.numpy().ravel()
    return g


def main():
    if SIM:
        cfg = dict(n_classes=5, img_size=8, c1=2, c2=3, hidden=6, batch_max=4)
        n_lote, tol = 4, 2e-3
        print("Modo SIMULADOR de Numba: red diminuta (solo valida la logica de los kernels).")
    else:
        cfg = dict(n_classes=29, img_size=64, c1=16, c2=32, hidden=128, batch_max=128)
        n_lote, tol = 32, 2e-3
        print("GPU real:", cuda.get_current_device().name)

    net = CudaSmallCNN(dropout=0.0, seed=7, **cfg)
    B, H, C = net.B, net.H1, net.C
    rng = np.random.default_rng(0)
    N = n_lote + 3
    X_host = rng.integers(0, 256, size=(N, H, H, C), dtype=np.uint8)
    y_host = rng.integers(0, net.K, size=N).astype(np.int32)
    X = cuda.to_device(X_host.reshape(-1))
    y = cuda.to_device(y_host)
    idx = cuda.to_device(np.arange(N, dtype=np.int32))
    preds = cuda.device_array(N, np.int32)

    print("\n[1-2] Forward y backward vs PyTorch (lote completo y lote parcial)")
    for n, etiqueta in ((n_lote, f"lote completo n={n_lote}"), (3, "lote parcial n=3")):
        off = 0 if n == n_lote else n_lote
        p = net.get_params()
        ref = modelo_referencia(net, p)
        xb = torch.from_numpy(X_host[off:off + n].astype(np.float64) / 255.0).permute(0, 3, 1, 2).contiguous()
        yb = torch.from_numpy(y_host[off:off + n].astype(np.int64))
        logits_ref = ref(xb)
        loss_ref = nn.functional.cross_entropy(logits_ref, yb)
        loss_ref.backward()
        g_ref = grads_referencia(ref)

        net._reset_stats()
        net.adelante(n, False, 0, X, idx, off)
        net._perdida(n, y, idx, off, preds, 0)
        net.atras(n)
        cuda.synchronize()
        logits = net.z2.copy_to_host()[: n * net.K].reshape(n, net.K) + p["bf2"]
        loss = float(net.stats.copy_to_host()[0]) / n
        check(f"logits ({etiqueta})", rel_err(logits, logits_ref.detach().numpy()) < tol,
              f"err_rel={rel_err(logits, logits_ref.detach().numpy()):.2e}")
        check(f"perdida ({etiqueta})", abs(loss - float(loss_ref)) < tol * max(1, abs(float(loss_ref))),
              f"cuda={loss:.5f} torch={float(loss_ref):.5f}")
        g = net.get_grads()
        for k in NOMBRES_PARAMS:
            e = rel_err(g[k], g_ref[k])
            check(f"grad {k:3s} ({etiqueta})", e < tol, f"err_rel={e:.2e}")

    print("\n[3] Un paso de Adam (lr=1e-3, weight_decay=1e-4) vs torch.optim.Adam")
    net.reiniciar(7)
    p0 = net.get_params()
    ref = modelo_referencia(net, p0)
    opt = torch.optim.Adam(ref.parameters(), lr=1e-3, weight_decay=1e-4)
    xb = torch.from_numpy(X_host[:n_lote].astype(np.float64) / 255.0).permute(0, 3, 1, 2).contiguous()
    yb = torch.from_numpy(y_host[:n_lote].astype(np.int64))
    opt.zero_grad(); nn.functional.cross_entropy(ref(xb), yb).backward(); opt.step()
    net.paso_entrenamiento(n_lote, X, y, idx, 0, 1e-3, 1e-4, 0, preds)
    cuda.synchronize()
    p1 = net.get_params()
    for k, (cap, attr) in zip(NOMBRES_PARAMS, [(0, "weight"), (0, "bias"), (3, "weight"), (3, "bias"),
                                               (7, "weight"), (7, "bias"), (9, "weight"), (9, "bias")]):
        # se compara el CAMBIO de parametro (lo que realmente hizo Adam)
        d_cuda = p1[k] - p0[k]
        d_ref = getattr(ref[cap], attr).detach().numpy().ravel() - p0[k].astype(np.float64)
        check(f"delta Adam {k:3s}", rel_err(d_cuda, d_ref) < 5e-2, f"err_rel={rel_err(d_cuda, d_ref):.2e}")

    print("\n[4] Dropout (p=0.3)")
    net_d = CudaSmallCNN(dropout=0.3, seed=1, **cfg)
    net_d.adelante(n_lote, True, 12345, X, idx, 0)
    mk = net_d.mask.copy_to_host()[: n_lote * net_d.hidden]
    keep = float(np.mean(mk > 0))
    esperado = 0.7
    check("fraccion conservada ~ 1-p", abs(keep - esperado) < (0.25 if SIM else 0.05), f"{keep:.3f}")
    check("escala 1/(1-p)", np.allclose(mk[mk > 0], 1 / 0.7, rtol=1e-5))
    net_d.adelante(n_lote, False, 12345, X, idx, 0)
    check("evaluacion: mascara = 1", np.all(net_d.mask.copy_to_host()[: n_lote * net_d.hidden] == 1.0))

    print("\n[5] Parametros vs models.SmallCNN")
    if not SIM:
        import models as mdl
        n_ref = sum(p.numel() for p in mdl.SmallCNN(29, 64).parameters())
        check("n_parametros == models.SmallCNN", net.n_parametros == n_ref, f"{net.n_parametros:,} vs {n_ref:,}")
    else:
        print("  (omitido en simulador: la red de prueba es mas chica)")

    print("\n[6] La perdida baja al entrenar")
    net_t = CudaSmallCNN(dropout=0.0, seed=3, **cfg)
    losses = []
    for _ in range(8 if SIM else 40):
        net_t._reset_stats()
        net_t.paso_entrenamiento(n_lote, X, y, idx, 0, 5e-3, 0.0, 0, preds)
        losses.append(float(net_t.stats.copy_to_host()[0]) / n_lote)
    check("perdida final < perdida inicial", losses[-1] < losses[0], f"{losses[0]:.3f} -> {losses[-1]:.3f}")

    print()
    if fallos:
        print(f"VERIFICACION FALLIDA ({len(fallos)}): {fallos}")
        return 1
    print("VERIFICACION EXITOSA: los kernels CUDA coinciden con la referencia de PyTorch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
