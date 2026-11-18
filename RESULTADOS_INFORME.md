# Resultados medidos del Proyecto 3

# Resultados: secuencial vs CUDA

Repeticiones: secuencial = 3, CUDA = 3 (media ± desviación estándar muestral).

## Tabla 1. Comparativa de tiempos y speedup por fase

| Fase | Secuencial (s) | CUDA (s) | Speedup |
|---|---:|---:|---:|
| Preparación de datos | 15.57 ± 0.12 | 8.68 ± 0.06 | 1.79× |
| · carga + decodificación + resize | 8.80 ± 0.21 | 1.86 ± 0.02 | 4.72× |
| · partición estratificada | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.02× |
| · guardado .npz | 5.97 ± 0.09 | 6.03 ± 0.05 | 0.99× |
| · lectura .npz | 0.79 ± 0.02 | 0.79 ± 0.01 | 1.01× |
| Transferencia CPU→GPU (H2D) | — | 0.05 ± 0.00 | — |
| Entrenamiento | 200.57 ± 3.75 | 21.01 ± 0.58 | 9.55× |
| Evaluación en test | 1.57 ± 0.02 | 0.04 ± 0.00 | 36.98× |
| **TOTAL del pipeline** | 217.71 ± 3.67 | 29.78 ± 0.59 | 7.31× |
| **TOTAL sin preparación de datos** | 202.14 ± 3.77 | 21.10 ± 0.58 | 9.58× |

_La compilación JIT de los kernels (una sola vez por proceso, fuera de los totales) tomó 1.46 s._

## Tabla 2. Tiempo por época de entrenamiento

| | Secuencial | CUDA | Speedup |
|---|---:|---:|---:|
| s/época (media ± desv.) | 13.31 ± 0.25 | 1.400 ± 0.039 | 9.5× |

## Tabla 3. Calidad del modelo (misma arquitectura, mismos datos)

| Métrica | Secuencial (PyTorch CPU) | CUDA (kernels propios) |
|---|---:|---:|
| Accuracy de validación (última época) | 84.25% | 82.18% |
| Accuracy en test | 84.83% | 82.34% |

## Tabla 4. Hardware y condiciones

- **Secuencial**: Windows 11 · CPU AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD · 8 núcleos físicos / 16 lógicos · RAM 31.1 GB · procesos de preparación: 1 · hilos PyTorch: 1
- **CUDA**: Windows 11 · CPU AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD · 8 núcleos físicos / 16 lógicos · RAM 31.1 GB · procesos de preparación: 16
  - GPU: NVIDIA GeForce RTX 4070 Ti · cc [8, 9] · 60 SM · 12282 MB

## Tabla 5. Uso de recursos — versión secuencial

| Fase | CPU medio (%) | CPU máx (%) | Núcleos con >50% medio | GPU util. media (%) | GPU util. máx (%) | VRAM máx (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Preparación (CPU) | 12 | 23 | 0/16 | 33 | 36 | 2638 |
| Entrenamiento | 12 | 22 | 0/16 | 33 | 53 | 2642 |
| Evaluación | 12 | 14 | 0/16 | 33 | 34 | 2626 |

## Tabla 6. Uso de recursos — versión CUDA

| Fase | CPU medio (%) | CPU máx (%) | Núcleos con >50% medio | GPU util. media (%) | GPU util. máx (%) | VRAM máx (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Preparación (CPU) | 22 | 98 | 1/16 | 0 | 1 | 2957 |
| Entrenamiento | 8 | 18 | 0/16 | 100 | 100 | 3334 |
| Evaluación | 10 | 10 | 2/16 | 99 | 99 | 3319 |

## Verificación de kernels contra PyTorch

```
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 16 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 16 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 4 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 1 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 1 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 16 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 32 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 16 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\pruebas\verificar_kernels.py:118: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\autograd\generated\python_variable_methods.cpp:837.)
  check(f"perdida ({etiqueta})", abs(loss - float(loss_ref)) < tol * max(1, abs(float(loss_ref))),
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 96 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 8 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 2 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 2 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 1 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 18 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
C:\Users\cr57b\Documentos\Proyecto_3_deep_learning\.venv\Lib\site-packages\numba_cuda\numba\cuda\dispatcher.py:748: NumbaPerformanceWarning: [1mGrid size 15 will likely result in GPU under-utilization due to low occupancy.[0m
  warn(errors.NumbaPerformanceWarning(msg))
GPU real: NVIDIA GeForce RTX 4070 Ti

[1-2] Forward y backward vs PyTorch (lote completo y lote parcial)
  [OK ] logits (lote completo n=32) err_rel=9.60e-07
  [OK ] perdida (lote completo n=32) cuda=3.38783 torch=3.38783
  [OK ] grad w1  (lote completo n=32) err_rel=3.65e-07
  [OK ] grad b1  (lote completo n=32) err_rel=2.82e-07
  [OK ] grad w2  (lote completo n=32) err_rel=3.07e-07
  [OK ] grad b2  (lote completo n=32) err_rel=1.98e-07
  [OK ] grad wf1 (lote completo n=32) err_rel=4.23e-07
  [OK ] grad bf1 (lote completo n=32) err_rel=1.72e-07
  [OK ] grad wf2 (lote completo n=32) err_rel=1.03e-06
  [OK ] grad bf2 (lote completo n=32) err_rel=8.00e-08
  [OK ] logits (lote parcial n=3) err_rel=7.93e-07
  [OK ] perdida (lote parcial n=3) cuda=3.37858 torch=3.37858
  [OK ] grad w1  (lote parcial n=3) err_rel=2.23e-07
  [OK ] grad b1  (lote parcial n=3) err_rel=3.72e-07
  [OK ] grad w2  (lote parcial n=3) err_rel=2.62e-07
  [OK ] grad b2  (lote parcial n=3) err_rel=2.39e-07
  [OK ] grad wf1 (lote parcial n=3) err_rel=3.55e-07
  [OK ] grad bf1 (lote parcial n=3) err_rel=1.09e-07
  [OK ] grad wf2 (lote parcial n=3) err_rel=3.53e-06
  [OK ] grad bf2 (lote parcial n=3) err_rel=8.98e-08

[3] Un paso de Adam (lr=1e-3, weight_decay=1e-4) vs torch.optim.Adam
  [OK ] delta Adam w1  err_rel=6.64e-05
  [OK ] delta Adam b1  err_rel=7.71e-06
  [OK ] delta Adam w2  err_rel=9.12e-05
  [OK ] delta Adam b2  err_rel=1.05e-05
  [OK ] delta Adam wf1 err_rel=1.87e-02
  [OK ] delta Adam bf1 err_rel=7.29e-06
  [OK ] delta Adam wf2 err_rel=1.15e-02
  [OK ] delta Adam bf2 err_rel=9.76e-06

[4] Dropout (p=0.3)
  [OK ] fraccion conservada ~ 1-p 0.694
  [OK ] escala 1/(1-p) 
  [OK ] evaluacion: mascara = 1 

[5] Parametros vs models.SmallCNN
  [OK ] n_parametros == models.SmallCNN 1,057,533 vs 1,057,533

[6] La perdida baja al entrenar
  [OK ] perdida final < perdida inicial 3.365 -> 0.017

VERIFICACION EXITOSA: los kernels CUDA coinciden con la referencia de PyTorch.
```

## Perfil secuencial

# Analisis de la solucion secuencial (CPU)

Parametros: 17,400 imagenes (600/clase, 29 clases), 64x64, batch 128, 15 epocas, SmallCNN (1,057,533 parametros), 1 hilo(s) de PyTorch.

## Tabla 1. Operaciones identificadas (de la imagen al resultado)

| No. | Operacion identificada | Descripcion de su funcionamiento |
|---|---|---|
| 1 | Listado y submuestreo | Se listan los .jpg de cada clase y se eligen al azar `n_per_class` por clase (rng con semilla). |
| 2 | Lectura + decodificacion JPG | Se lee el archivo y se decodifica a pixeles RGB 200x200 (CPU, acceso a disco). |
| 3 | Redimensionado | Interpolacion bilineal 200x200 -> 64x64 por imagen. |
| 4 | Particion estratificada | Indices train/val/test 70/15/15 por clase; se guardan en .npz. |
| 5 | Armado de lote | `Dataset.__getitem__`: uint8 HWC -> float32 CHW /255; `DataLoader` mezcla y apila 128 imagenes. |
| 6 | Forward de la CNN | Conv3x3(3->16)+ReLU+MaxPool, Conv3x3(16->32)+ReLU+MaxPool, Linear(8192->128)+ReLU+Dropout, Linear(128->29). |
| 7 | Perdida | Entropia cruzada (softmax + log-verosimilitud) promediada sobre el lote. |
| 8 | Backward | Retropropagacion: gradientes de convoluciones, capas lineales, ReLU, pooling y dropout. |
| 9 | Actualizacion Adam | Momentos de primer/segundo orden y actualizacion de los 8 tensores de parametros. |
| 10 | Evaluacion | Forward sin dropout sobre validacion (cada epoca) y test; argmax y accuracy. |

## Tabla 2. Operaciones repetitivas y cantidad de repeticiones

| Operacion repetitiva | Datos que procesa | Cantidad de repeticiones |
|---|---|---|
| Lectura + decodificacion + resize | 1 imagen JPG 200x200 | 17,400 (una vez por imagen de la submuestra; el dataset completo tiene 87,000) |
| Armado de lote (/255, CHW) | 1 imagen 64x64x3 | 224,460 = (12,180 train + 2,610 val) x 15 epocas + 2,610 test |
| Convolucion 3x3 + ReLU + MaxPool (x2 capas) | lote [128,3,64,64] | forward: 1,440 lotes de train + 315 de val + 21 de test; backward: 1,440 |
| Capas lineales (matmul) | [128,8192] x [8192,128] y [128,128] x [128,29] | igual que arriba (forward) y 2 matmul extra por capa en backward |
| Producto-acumulacion (MAC) por imagen | 7,540,352 MAC forward (conv1 1,769,472; conv2 4,718,592; fc1 1,048,576; fc2 3,712) | 182,700 imagenes de train x ~3 (fwd+bwd) = 4.133e+12 MAC |
| Actualizacion Adam | 8 tensores, 1,057,533 parametros | 1,440 pasos |

## Tabla 3. Tiempo medido por imagen en la preparacion de datos (CPU, 1 proceso)

| Etapa | ms por imagen | % |
|---|---|---|
| leer_archivo | 0.105 | 18.7% |
| decodificar_jpg_a_rgb | 0.318 | 56.5% |
| resize_bilineal | 0.140 | 24.8% |
| **Total por imagen** | **0.563** | 100% |

Proyeccion para 17,400 imagenes en un solo proceso: **9.8 s** (sin contar particion ni guardado).

## Tabla 4. Desglose medido de un paso de entrenamiento (CPU)

| Etapa | ms por lote | % |
|---|---|---|
| armar_lote (Dataset+DataLoader: /255, permutar a CHW, apilar) | 3.50 | 2.8% |
| forward (conv+ReLU+pool+lineal) | 77.27 | 62.1% |
| perdida (entropia cruzada) | 0.08 | 0.1% |
| backward (autograd) | 40.98 | 32.9% |
| optimizador Adam | 2.58 | 2.1% |
| **Total por lote** | **124.41** | 100% |

Proyeccion de entrenamiento: 179.2 s (1,440 pasos), sin contar la validacion de cada epoca.

## Tabla 5. Operadores que mas tiempo de CPU consumen (torch.profiler)

| Operador | Llamadas | CPU propio (ms) | % |
|---|---|---|---|
| `aten::max_pool2d_with_indices` | 16 | 510.7 | 47.0% |
| `aten::convolution_backward` | 16 | 216.0 | 19.9% |
| `aten::mkldnn_convolution` | 16 | 140.8 | 12.9% |
| `aten::mm` | 32 | 44.8 | 4.1% |
| `aten::threshold_backward` | 24 | 28.9 | 2.7% |
| `aten::max_pool2d_with_indices_backward` | 16 | 24.4 | 2.2% |
| `aten::addmm` | 16 | 21.7 | 2.0% |
| `aten::sqrt` | 64 | 14.0 | 1.3% |

## Entorno

```
{
  "python": "3.13.7",
  "sistema": "Windows 11",
  "procesador": "AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD",
  "nucleos_logicos": 16,
  "nucleos_fisicos": 8,
  "ram_total_gb": 31.1,
  "gpu_nvidia_smi": {
    "nombre": "NVIDIA GeForce RTX 4070 Ti",
    "driver": "610.88",
    "memoria_total_mb": "12282"
  },
  "torch": {
    "version": "2.11.0+cpu",
    "hilos_cpu_por_defecto": 8,
    "cuda_en_torch": false
  },
  "numba": {
    "version": "0.68.1",
    "cuda_disponible": true
  },
  "gpu_numba": {
    "nombre": "NVIDIA GeForce RTX 4070 Ti",
    "compute_capability": [
      8,
      9
    ],
    "multiprocesadores": 60,
    "max_hilos_por_bloque": 1024,
    "memoria_total_mb": 12282,
    "memoria_libre_mb": 11056
  },
  "cuda_disponible": true
}
```
