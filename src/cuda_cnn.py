"""SmallCNN del Lab 3 entrenada en GPU con kernels CUDA propios (ver cuda_kernels.py).

Arquitectura (identica a models.SmallCNN, 1,057,533 parametros con los valores por defecto):
    Conv3x3(3->16)+ReLU -> MaxPool2 -> Conv3x3(16->32)+ReLU -> MaxPool2
    -> Flatten -> Linear(8192->128)+ReLU -> Dropout(0.3) -> Linear(128->29)

Esta clase NO importa PyTorch. Solo orquesta lanzamientos de kernels sobre buffers que se
reservan UNA vez (sin cudaMalloc dentro del bucle de entrenamiento).
"""
import time

import numpy as np
from numba import cuda

from cuda_kernels import (
    RED, TILE, TPB,
    k_adam, k_bias_relu_dropout, k_conv3x3_bwd_entrada, k_conv3x3_bwd_pesos, k_conv3x3_relu,
    k_conv_bwd_sesgo, k_dropout_relu_bwd, k_gemm, k_maxpool2, k_pool_relu_bwd,
    k_preparar_lote, k_softmax_ce, k_suma_columnas,
)

NOMBRES_PARAMS = ["w1", "b1", "w2", "b2", "wf1", "bf1", "wf2", "bf2"]


def _g1(total):
    """Bloques necesarios para cubrir 'total' hilos 1-D."""
    return (int(total) + TPB - 1) // TPB


def _grid_conv(W, H, z):
    return ((W + TILE - 1) // TILE, (H + TILE - 1) // TILE, int(z))


class CudaSmallCNN:
    def __init__(self, n_classes=29, img_size=64, c_in=3, c1=16, c2=32, hidden=128,
                 dropout=0.3, batch_max=128, seed=42):
        assert img_size % 4 == 0, "img_size debe ser multiplo de 4 (dos max-pool 2x2)"
        assert batch_max <= 1024, "batch_max > 1024 no cabe en un bloque de k_softmax_ce"
        self.K, self.C, self.H1, self.H2, self.H3 = n_classes, c_in, img_size, img_size // 2, img_size // 4
        self.c1, self.c2, self.hidden = c1, c2, hidden
        self.F = c2 * self.H3 * self.H3
        self.p_drop = float(dropout)
        self.B = int(batch_max)
        self.seed = seed

        f32 = np.float32
        B, H1, H2, H3, F = self.B, self.H1, self.H2, self.H3, self.F
        self.tamanos = {
            "w1": c1 * c_in * 9, "b1": c1, "w2": c2 * c1 * 9, "b2": c2,
            "wf1": hidden * F, "bf1": hidden, "wf2": n_classes * hidden, "bf2": n_classes,
        }
        # parametros, gradientes y estados de Adam (todo en memoria de la GPU)
        self.P = {k: cuda.device_array(v, f32) for k, v in self.tamanos.items()}
        self.G = {k: cuda.device_array(v, f32) for k, v in self.tamanos.items()}
        self.M = {k: cuda.device_array(v, f32) for k, v in self.tamanos.items()}
        self.V = {k: cuda.device_array(v, f32) for k, v in self.tamanos.items()}
        # activaciones y gradientes de activaciones (reservados una sola vez)
        d = lambda n_: cuda.device_array(int(n_), f32)  # noqa: E731
        self.xb = d(B * c_in * H1 * H1)
        self.a1, self.p1 = d(B * c1 * H1 * H1), d(B * c1 * H2 * H2)
        self.a2, self.p2 = d(B * c2 * H2 * H2), d(B * F)
        self.arg1 = cuda.device_array(B * c1 * H2 * H2, np.int32)
        self.arg2 = cuda.device_array(B * F, np.int32)
        self.z1, self.h, self.dr, self.mask = d(B * hidden), d(B * hidden), d(B * hidden), d(B * hidden)
        self.z2, self.dlog = d(B * n_classes), d(B * n_classes)
        self.dd, self.dz1 = d(B * hidden), d(B * hidden)
        self.dp2, self.dza2 = d(B * F), d(B * c2 * H2 * H2)
        self.dp1, self.dza1 = d(B * c1 * H2 * H2), d(B * c1 * H1 * H1)
        self.stats = cuda.to_device(np.zeros(2, f32))
        self.reiniciar(seed)

    # ------------------------------------------------------------------ parametros
    def _init_numpy(self, seed):
        """Inicializacion por defecto de PyTorch (kaiming_uniform a=sqrt(5) => U(-1/sqrt(fan_in), +))."""
        rng = np.random.default_rng(seed)

        def U(fan_in, size):
            b = 1.0 / np.sqrt(fan_in)
            return rng.uniform(-b, b, size).astype(np.float32)

        t = self.tamanos
        return {
            "w1": U(self.C * 9, t["w1"]), "b1": U(self.C * 9, t["b1"]),
            "w2": U(self.c1 * 9, t["w2"]), "b2": U(self.c1 * 9, t["b2"]),
            "wf1": U(self.F, t["wf1"]), "bf1": U(self.F, t["bf1"]),
            "wf2": U(self.hidden, t["wf2"]), "bf2": U(self.hidden, t["bf2"]),
        }

    def set_params(self, params):
        for k in NOMBRES_PARAMS:
            self.P[k].copy_to_device(np.ascontiguousarray(params[k], dtype=np.float32).reshape(-1))

    def get_params(self):
        return {k: self.P[k].copy_to_host() for k in NOMBRES_PARAMS}

    def get_grads(self):
        return {k: self.G[k].copy_to_host() for k in NOMBRES_PARAMS}

    def reiniciar(self, seed=None):
        """Pesos nuevos + estados de Adam en cero."""
        self.set_params(self._init_numpy(self.seed if seed is None else seed))
        for k, n in self.tamanos.items():
            self.M[k].copy_to_device(np.zeros(n, np.float32))
            self.V[k].copy_to_device(np.zeros(n, np.float32))
        self.paso_adam = 0

    @property
    def n_parametros(self):
        return int(sum(self.tamanos.values()))

    # ------------------------------------------------------------------ propagacion
    def _gemm(self, a, b, c, M, N, K, sa0, sa1, sb0, sb1):
        k_gemm[((N + TILE - 1) // TILE, (M + TILE - 1) // TILE), (TILE, TILE)](
            a, b, c, M, N, K, sa0, sa1, sb0, sb1)

    def adelante(self, n, entrenando, semilla=0, X=None, idx=None, off=0):
        """Forward del lote. Si X no es None, primero arma el lote desde la imagen cruda en GPU."""
        P = self.P
        C, c1, c2, H1, H2, F, hid, K = self.C, self.c1, self.c2, self.H1, self.H2, self.F, self.hidden, self.K
        if X is not None:
            k_preparar_lote[_g1(n * C * H1 * H1), TPB](X, idx, off, self.xb, n, C, H1, H1)
        k_conv3x3_relu[_grid_conv(H1, H1, n * c1), (TILE, TILE)](self.xb, P["w1"], P["b1"], self.a1, n, C, c1, H1, H1)
        k_maxpool2[_g1(n * c1 * H2 * H2), TPB](self.a1, self.p1, self.arg1, n, c1, H1, H1)
        k_conv3x3_relu[_grid_conv(H2, H2, n * c2), (TILE, TILE)](self.p1, P["w2"], P["b2"], self.a2, n, c1, c2, H2, H2)
        k_maxpool2[_g1(n * F), TPB](self.a2, self.p2, self.arg2, n, c2, H2, H2)
        # z1 = p2 @ Wf1^T
        self._gemm(self.p2, P["wf1"], self.z1, n, hid, F, F, 1, 1, F)
        k_bias_relu_dropout[_g1(n * hid), TPB](self.z1, P["bf1"], self.h, self.dr, self.mask, n, hid,
                                               int(self.p_drop * 4294967296.0), 1.0 / (1.0 - self.p_drop), semilla,
                                               1 if (entrenando and self.p_drop > 0) else 0)
        # z2 = dr @ Wf2^T   (el sesgo bf2 se suma dentro de k_softmax_ce)
        self._gemm(self.dr, P["wf2"], self.z2, n, K, hid, hid, 1, 1, hid)

    def atras(self, n):
        """Backward: rellena self.G con los gradientes de todos los parametros."""
        P, G = self.P, self.G
        C, c1, c2, H1, H2, F, hid, K = self.C, self.c1, self.c2, self.H1, self.H2, self.F, self.hidden, self.K
        # capa final: dWf2 = dlog^T @ dr ; dbf2 = sum(dlog) ; dd = dlog @ Wf2
        self._gemm(self.dlog, self.dr, G["wf2"], K, hid, n, 1, K, hid, 1)
        k_suma_columnas[_g1(K), TPB](self.dlog, G["bf2"], n, K)
        self._gemm(self.dlog, P["wf2"], self.dd, n, hid, K, K, 1, hid, 1)
        k_dropout_relu_bwd[_g1(n * hid), TPB](self.dd, self.mask, self.h, self.dz1, n * hid)
        # capa densa 1: dWf1 = dz1^T @ p2 ; dbf1 ; dp2 = dz1 @ Wf1
        self._gemm(self.dz1, self.p2, G["wf1"], hid, F, n, 1, hid, F, 1)
        k_suma_columnas[_g1(hid), TPB](self.dz1, G["bf1"], n, hid)
        self._gemm(self.dz1, P["wf1"], self.dp2, n, F, hid, hid, 1, F, 1)
        # pool2 + relu2 -> conv2
        k_pool_relu_bwd[_g1(n * c2 * H2 * H2), TPB](self.dp2, self.arg2, self.a2, self.dza2, n, c2, H2, H2)
        k_conv3x3_bwd_pesos[c2 * c1 * 9, RED](self.p1, self.dza2, G["w2"], n, c1, c2, H2, H2)
        k_conv_bwd_sesgo[c2, RED](self.dza2, G["b2"], n, c2, H2 * H2)
        k_conv3x3_bwd_entrada[_grid_conv(H2, H2, n * c1), (TILE, TILE)](self.dza2, P["w2"], self.dp1, n, c1, c2, H2, H2)
        # pool1 + relu1 -> conv1 (la imagen no necesita gradiente)
        k_pool_relu_bwd[_g1(n * c1 * H1 * H1), TPB](self.dp1, self.arg1, self.a1, self.dza1, n, c1, H1, H1)
        k_conv3x3_bwd_pesos[c1 * C * 9, RED](self.xb, self.dza1, G["w1"], n, C, c1, H1, H1)
        k_conv_bwd_sesgo[c1, RED](self.dza1, G["b1"], n, c1, H1 * H1)

    def _adam(self, lr, wd, beta1=0.9, beta2=0.999, eps=1e-8):
        self.paso_adam += 1
        t = self.paso_adam
        bc1, bc2 = 1.0 - beta1 ** t, 1.0 - beta2 ** t
        for k in NOMBRES_PARAMS:
            size = self.tamanos[k]
            k_adam[_g1(size), TPB](self.P[k], self.G[k], self.M[k], self.V[k], size,
                                   lr, beta1, beta2, eps, wd, bc1, bc2)

    def _perdida(self, n, labels, idx, off, preds, poff):
        k_softmax_ce[1, max(32, ((n + 31) // 32) * 32)](self.z2, self.P["bf2"], labels, idx, off, self.dlog,
                                                       self.stats, preds, poff, n, self.K)

    def paso_entrenamiento(self, n, X, labels, idx, off, lr, wd, semilla, preds):
        """Un paso completo: lote -> forward -> perdida -> backward -> Adam (todo en GPU)."""
        self.adelante(n, True, semilla, X, idx, off)
        self._perdida(n, labels, idx, off, preds, 0)
        self.atras(n)
        self._adam(lr, wd)

    # ------------------------------------------------------------------ bucles
    def _reset_stats(self):
        self.stats.copy_to_device(np.zeros(2, np.float32))

    def evaluar(self, X, labels, n_total):
        """Forward en modo evaluacion (sin dropout). Devuelve (perdida, accuracy, predicciones)."""
        idx = cuda.to_device(np.arange(n_total, dtype=np.int32))
        preds = cuda.device_array(n_total, np.int32)
        self._reset_stats()
        for off in range(0, n_total, self.B):
            n = min(self.B, n_total - off)
            self.adelante(n, False, 0, X, idx, off)
            self._perdida(n, labels, idx, off, preds, off)
        s = self.stats.copy_to_host()
        return float(s[0]) / n_total, float(s[1]) / n_total, preds.copy_to_host()

    def entrenar(self, X, labels, n_train, Xv, labels_v, n_val, epochs, lr, weight_decay,
                 batch_size=None, seed=42, log=print):
        """Entrena 'epochs' epocas; devuelve un historial con las mismas llaves que train_utils.train_model."""
        bs = min(batch_size or self.B, self.B)
        hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
        preds_dummy = cuda.device_array(max(bs, 1), np.int32)
        paso = 0
        for ep in range(epochs):
            cuda.synchronize()
            t0 = time.time()
            perm = np.random.default_rng(seed + ep).permutation(n_train).astype(np.int32)
            idx = cuda.to_device(perm)
            self._reset_stats()
            for off in range(0, n_train, bs):
                n = min(bs, n_train - off)
                paso += 1
                self.paso_entrenamiento(n, X, labels, idx, off, lr, weight_decay,
                                        (seed * 1000003 + paso) % (2 ** 31 - 1), preds_dummy)
            s = self.stats.copy_to_host()          # primer D2H de la epoca: sincroniza la GPU
            tr_loss, tr_acc = float(s[0]) / n_train, float(s[1]) / n_train
            va_loss, va_acc, _ = self.evaluar(Xv, labels_v, n_val)
            cuda.synchronize()
            dt = time.time() - t0
            for k, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc, dt)):
                hist[k].append(v)
            log(f"epoch {ep + 1}/{epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} ({dt:.1f}s)")
        return hist

    def calentar_jit(self):
        """Compila TODOS los kernels con un lote minusculo y restaura pesos y Adam.
        Asi la compilacion JIT (unica vez) queda fuera del tiempo de entrenamiento medido."""
        n = 2
        X = cuda.to_device(np.zeros(n * self.H1 * self.H1 * self.C, np.uint8))
        y = cuda.to_device(np.zeros(n, np.int32))
        idx = cuda.to_device(np.arange(n, dtype=np.int32))
        preds = cuda.device_array(n, np.int32)
        self.paso_entrenamiento(n, X, y, idx, 0, 1e-3, 1e-4, 1, preds)
        self.evaluar(X, y, n)
        cuda.synchronize()
        self.reiniciar(self.seed)
