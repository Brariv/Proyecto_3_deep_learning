"""Kernels CUDA escritos a mano (CUDA Python con Numba) para entrenar la SmallCNN del Lab 3.

NINGUNA operacion de la red usa PyTorch: cada capa (convolucion, ReLU, max-pool, capa
totalmente conectada, dropout, softmax + entropia cruzada, retropropagacion y el
optimizador Adam) esta implementada como un kernel @cuda.jit propio.

Convenciones
------------
* Todos los tensores son arreglos 1-D contiguos float32 y se indexan a mano
  (layout NCHW: indice = ((n*C + c)*H + y)*W + x).
* Pesos de convolucion: (cout, cin, 3, 3).  Pesos densos: (salida, entrada).
* "n" es el tamano REAL del lote (el ultimo lote de una epoca puede ser menor que batch_max).

Mapeo de trabajo a la GPU (resumen; el detalle esta en el notebook asl_cuda.ipynb)
-------------------------------------------------------------------------------
  k_preparar_lote        1 hilo = 1 valor de salida           grid 1D  x bloque 256
  k_conv3x3_relu         1 hilo = 1 pixel de salida           grid (W/16, H/16, n*cout) x bloque 16x16
  k_maxpool2             1 hilo = 1 valor de salida           grid 1D  x bloque 256
  k_gemm                 1 hilo = 1 elemento de C (tile 16x16 en memoria compartida)
  k_bias_relu_dropout    1 hilo = 1 activacion                grid 1D  x bloque 256
  k_softmax_ce           1 hilo = 1 muestra del lote          1 bloque
  k_pool_relu_bwd        1 hilo = 1 pixel de entrada del pool grid 1D  x bloque 256
  k_conv3x3_bwd_pesos    1 BLOQUE = 1 peso del kernel; 256 hilos reducen sobre n*H*W (memoria compartida)
  k_conv3x3_bwd_entrada  1 hilo = 1 pixel de entrada          grid (W/16, H/16, n*cin) x bloque 16x16
  k_adam                 1 hilo = 1 parametro                 grid 1D  x bloque 256
"""
import math

from numba import cuda, float32, int64

TILE = 16   # lado del tile de GEMM y del bloque 2-D de convolucion
RED = 256   # hilos por bloque en los kernels de reduccion
TPB = 256   # hilos por bloque en los kernels 1-D


# ----------------------------------------------------------------------------------------
# Entrada: uint8 NHWC (como sale de la preparacion en CPU)  ->  float32 NCHW en [0, 1]
# ----------------------------------------------------------------------------------------
@cuda.jit
def k_preparar_lote(x_u8, idx, off, out, n, C, H, W):
    """out[b,c,y,x] = x_u8[idx[off+b], y, x, c] / 255.  (gather del lote + normalizacion)"""
    i = cuda.grid(1)
    if i < n * C * H * W:
        xw = i % W
        t = i // W
        yh = t % H
        t = t // H
        c = t % C
        b = t // C
        s = int64(idx[off + b])
        out[i] = float32(x_u8[((s * H + yh) * W + xw) * C + c]) / float32(255.0)


# ----------------------------------------------------------------------------------------
# Convolucion 3x3 (padding 1) + ReLU
# ----------------------------------------------------------------------------------------
@cuda.jit
def k_conv3x3_relu(x, w, bias, out, n, cin, cout, H, W):
    """out[b,co,y,x] = ReLU(bias[co] + sum_{ci,ky,kx} x[b,ci,y+ky-1,x+kx-1] * w[co,ci,ky,kx])"""
    xo = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    yo = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    z = cuda.blockIdx.z                      # z = b * cout + co
    if xo < W and yo < H:
        b = z // cout
        co = z % cout
        acc = bias[co]
        for ci in range(cin):
            xbase = (b * cin + ci) * H * W
            wbase = (co * cin + ci) * 9
            for ky in range(3):
                yi = yo + ky - 1
                if yi >= 0 and yi < H:
                    for kx in range(3):
                        xi = xo + kx - 1
                        if xi >= 0 and xi < W:
                            acc += x[xbase + yi * W + xi] * w[wbase + ky * 3 + kx]
        if acc < float32(0.0):
            acc = float32(0.0)
        out[(z * H + yo) * W + xo] = acc


# ----------------------------------------------------------------------------------------
# Max-pool 2x2 (stride 2) guardando la posicion del maximo para la retropropagacion
# ----------------------------------------------------------------------------------------
@cuda.jit
def k_maxpool2(a, out, arg, n, c, H, W):
    i = cuda.grid(1)
    Ho = H // 2
    Wo = W // 2
    if i < n * c * Ho * Wo:
        xo = i % Wo
        t = i // Wo
        yo = t % Ho
        plano = t // Ho                       # plano = b * c + canal
        base = plano * H * W
        p = (2 * yo) * W + 2 * xo
        mejor = a[base + p]
        pos = p
        v = a[base + p + 1]
        if v > mejor:
            mejor = v
            pos = p + 1
        v = a[base + p + W]
        if v > mejor:
            mejor = v
            pos = p + W
        v = a[base + p + W + 1]
        if v > mejor:
            mejor = v
            pos = p + W + 1
        out[i] = mejor
        arg[i] = pos


# ----------------------------------------------------------------------------------------
# GEMM con tiles de 16x16 en memoria compartida:  C[M,N] = A[M,K] @ B[K,N]
# A[i,k] = a[i*sa0 + k*sa1],  B[k,j] = b[k*sb0 + j*sb1]   -> las transpuestas se logran
# solo cambiando los strides (no hay kernels de transposicion).
# ----------------------------------------------------------------------------------------
@cuda.jit
def k_gemm(a, b, c, M, N, K, sa0, sa1, sb0, sb1):
    tile_a = cuda.shared.array((TILE, TILE), float32)
    tile_b = cuda.shared.array((TILE, TILE), float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    fila = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx
    acc = float32(0.0)
    for t in range((K + TILE - 1) // TILE):
        ka = t * TILE + tx
        if fila < M and ka < K:
            tile_a[ty, tx] = a[fila * sa0 + ka * sa1]
        else:
            tile_a[ty, tx] = float32(0.0)
        kb = t * TILE + ty
        if kb < K and col < N:
            tile_b[ty, tx] = b[kb * sb0 + col * sb1]
        else:
            tile_b[ty, tx] = float32(0.0)
        cuda.syncthreads()
        for k in range(TILE):
            acc += tile_a[ty, k] * tile_b[k, tx]
        cuda.syncthreads()
    if fila < M and col < N:
        c[fila * N + col] = acc


# ----------------------------------------------------------------------------------------
# Dropout con generador pseudoaleatorio propio (hash PCG), determinista por (semilla, indice)
# ----------------------------------------------------------------------------------------
@cuda.jit(device=True)
def _hash32(semilla, i):
    """Entero pseudoaleatorio de 32 bits (hash PCG) a partir de (semilla, indice); solo aritmetica entera."""
    s = (int64(i) * 747796405 + int64(semilla) * 2891336453 + 2891336453) & 0xFFFFFFFF
    palabra = (((s >> ((s >> 28) + 4)) ^ s) * 277803737) & 0xFFFFFFFF
    return ((palabra >> 22) ^ palabra) & 0xFFFFFFFF


@cuda.jit
def k_bias_relu_dropout(z, bias, h, d, mask, n, F, umbral, escala, semilla, entrenando):
    """h = ReLU(z + bias);  d = h * mascara.  Se descarta la activacion si hash32 < umbral = p_drop * 2^32;
    las que sobreviven se escalan por 1/(1-p)."""
    i = cuda.grid(1)
    if i < n * F:
        v = z[i] + bias[i % F]
        if v < float32(0.0):
            v = float32(0.0)
        h[i] = v
        m = float32(1.0)
        if entrenando != 0:
            if _hash32(semilla, i) >= umbral:
                m = float32(escala)
            else:
                m = float32(0.0)
        mask[i] = m
        d[i] = v * m


# ----------------------------------------------------------------------------------------
# Softmax + entropia cruzada (media del lote) + gradiente de los logits + aciertos
# ----------------------------------------------------------------------------------------
@cuda.jit
def k_softmax_ce(z, bias, labels, idx, off, dlogits, stats, preds, poff, n, K):
    """1 hilo por muestra. stats[0] += perdida, stats[1] += aciertos, preds[poff+b] = argmax."""
    b = cuda.grid(1)
    if b < n:
        base = b * K
        y = labels[idx[off + b]]
        m = z[base] + bias[0]
        arg = 0
        for k in range(1, K):
            v = z[base + k] + bias[k]
            if v > m:
                m = v
                arg = k
        s = float32(0.0)
        for k in range(K):
            s += math.exp(z[base + k] + bias[k] - m)
        loss = math.log(s) - (z[base + y] + bias[y] - m)
        inv_n = float32(1.0) / float32(n)
        for k in range(K):
            p = math.exp(z[base + k] + bias[k] - m) / s
            if k == y:
                p = p - float32(1.0)
            dlogits[base + k] = p * inv_n
        cuda.atomic.add(stats, 0, loss)
        if arg == y:
            cuda.atomic.add(stats, 1, float32(1.0))
        preds[poff + b] = arg


# ----------------------------------------------------------------------------------------
# Retropropagacion
# ----------------------------------------------------------------------------------------
@cuda.jit
def k_suma_columnas(d, out, n, F):
    """out[f] = sum_b d[b*F + f]   (gradiente del sesgo de una capa densa)"""
    f = cuda.grid(1)
    if f < F:
        s = float32(0.0)
        for b in range(n):
            s += d[b * F + f]
        out[f] = s


@cuda.jit
def k_dropout_relu_bwd(dd, mask, h, dz, total):
    i = cuda.grid(1)
    if i < total:
        g = dd[i] * mask[i]
        if h[i] <= float32(0.0):
            g = float32(0.0)
        dz[i] = g


@cuda.jit
def k_pool_relu_bwd(dpool, arg, a, dz, n, c, H, W):
    """Retropropaga a traves de max-pool 2x2 y de la ReLU anterior (version 'gather':
    cada hilo mira su pixel de entrada, no hay condiciones de carrera ni atomicas)."""
    i = cuda.grid(1)
    if i < n * c * H * W:
        xw = i % W
        t = i // W
        yh = t % H
        plano = t // H
        Ho = H // 2
        Wo = W // 2
        j = (plano * Ho + yh // 2) * Wo + xw // 2
        g = float32(0.0)
        if arg[j] == yh * W + xw and a[i] > float32(0.0):
            g = dpool[j]
        dz[i] = g


@cuda.jit
def k_conv3x3_bwd_pesos(x, dz, dw, n, cin, cout, H, W):
    """dw[co,ci,ky,kx] = sum_{b,y,x} dz[b,co,y,x] * x[b,ci,y+ky-1,x+kx-1].
    Un bloque por peso; sus 256 hilos reparten n*H*W terminos y los reducen en memoria compartida."""
    sm = cuda.shared.array(RED, float32)
    tid = cuda.threadIdx.x
    widx = cuda.blockIdx.x                   # = ((co*cin + ci)*3 + ky)*3 + kx
    kx = widx % 3
    t = widx // 3
    ky = t % 3
    t = t // 3
    ci = t % cin
    co = t // cin
    acc = float32(0.0)
    total = n * H * W
    i = tid
    while i < total:
        xo = i % W
        r = i // W
        yo = r % H
        b = r // H
        yi = yo + ky - 1
        xi = xo + kx - 1
        if yi >= 0 and yi < H and xi >= 0 and xi < W:
            acc += dz[((b * cout + co) * H + yo) * W + xo] * x[((b * cin + ci) * H + yi) * W + xi]
        i += RED
    sm[tid] = acc
    cuda.syncthreads()
    s = RED // 2
    while s > 0:
        if tid < s:
            sm[tid] += sm[tid + s]
        cuda.syncthreads()
        s //= 2
    if tid == 0:
        dw[widx] = sm[0]


@cuda.jit
def k_conv_bwd_sesgo(dz, db, n, cout, HW):
    """db[co] = sum_{b,pixel} dz[b,co,pixel]  (un bloque por canal de salida)."""
    sm = cuda.shared.array(RED, float32)
    tid = cuda.threadIdx.x
    co = cuda.blockIdx.x
    acc = float32(0.0)
    total = n * HW
    i = tid
    while i < total:
        b = i // HW
        r = i % HW
        acc += dz[(b * cout + co) * HW + r]
        i += RED
    sm[tid] = acc
    cuda.syncthreads()
    s = RED // 2
    while s > 0:
        if tid < s:
            sm[tid] += sm[tid + s]
        cuda.syncthreads()
        s //= 2
    if tid == 0:
        db[co] = sm[0]


@cuda.jit
def k_conv3x3_bwd_entrada(dz, w, dx, n, cin, cout, H, W):
    """dx[b,ci,y,x] = sum_{co,ky,kx} dz[b,co,y-ky+1,x-kx+1] * w[co,ci,ky,kx]"""
    xi = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    yi = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    z = cuda.blockIdx.z                      # z = b * cin + ci
    if xi < W and yi < H:
        b = z // cin
        ci = z % cin
        acc = float32(0.0)
        for co in range(cout):
            zbase = (b * cout + co) * H * W
            wbase = (co * cin + ci) * 9
            for ky in range(3):
                yo = yi - ky + 1
                if yo >= 0 and yo < H:
                    for kx in range(3):
                        xo = xi - kx + 1
                        if xo >= 0 and xo < W:
                            acc += dz[zbase + yo * W + xo] * w[wbase + ky * 3 + kx]
        dx[(z * H + yi) * W + xi] = acc


# ----------------------------------------------------------------------------------------
# Optimizador Adam (misma formulacion que torch.optim.Adam con weight_decay = L2)
# ----------------------------------------------------------------------------------------
@cuda.jit
def k_adam(p, g, m, v, size, lr, beta1, beta2, eps, wd, bc1, bc2):
    i = cuda.grid(1)
    if i < size:
        b1 = float32(beta1)
        b2 = float32(beta2)
        gi = g[i] + float32(wd) * p[i]
        mi = b1 * m[i] + (float32(1.0) - b1) * gi
        vi = b2 * v[i] + (float32(1.0) - b2) * gi * gi
        m[i] = mi
        v[i] = vi
        denom = math.sqrt(vi) / math.sqrt(float32(bc2)) + float32(eps)
        p[i] = p[i] - (float32(lr) / float32(bc1)) * mi / denom


KERNELS = [
    k_preparar_lote, k_conv3x3_relu, k_maxpool2, k_gemm, k_bias_relu_dropout, k_softmax_ce,
    k_suma_columnas, k_dropout_relu_bwd, k_pool_relu_bwd, k_conv3x3_bwd_pesos,
    k_conv_bwd_sesgo, k_conv3x3_bwd_entrada, k_adam,
]
