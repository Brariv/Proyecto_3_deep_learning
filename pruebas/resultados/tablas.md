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
