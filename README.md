# Actividad 3 — INFO1194

Algoritmo Genético Paralelo con CUDA para el Problema de la Mochila Extendida.

Tres variantes:
- **V1 — Secuencial (CPU)**: implementación clásica en CPU, línea base para speed-up.
- **V2 — CUDA Básico**: población en GPU, kernels paralelos, atomicCAS, `cub::DeviceRadixSort`.
- **V3 — CUDA Optimizado**: memoria constante, reducción en shared memory, elitismo GPU-side, sin divergencia de warps.

---

## Requisitos

- **CUDA Toolkit** 12.x (recomendado 12.4)
- **CMake** ≥ 3.25
- **Compilador** con soporte C++23 (g++ ≥ 13)
- **OpenMP** (paralelismo CPU en V1)
- **GPU** con compute capability ≥ 5.2 (para V2/V3)

### Instalación (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y build-essential cmake git libomp-dev
```

CUDA Toolkit se instala desde NVIDIA:  
<https://developer.nvidia.com/cuda-downloads>

---

## Compilación

### Con CMake presets (recomendado)

```bash
cmake --preset dev
cmake --build --preset dev
```

### Con Makefile de soporte

```bash
make build          # configura + compila
```

### Manual (cmake directo)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

El binario generado es `build/mochila_ga_cuda`.

---

## Variantes disponibles

| Flag `--variant` | Descripción |
|-----------------|-------------|
| `sequential`    | V1 — CPU secuencial |
| `cuda_basic`    | V2 — CUDA básico |
| `cuda_optimized`| V3 — CUDA optimizado |

---

## Argumentos de línea de comandos

| Argumento | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `--instance <path>` | Ruta al directorio de la instancia (obligatorio) | — |
| `--variant <name>` | Variante del algoritmo | `sequential` |
| `--pop-size <n>` | Tamaño de la población | `1024` |
| `--generations <n>` | Máximo de generaciones | `200` |
| `--seed <n>` | Semilla del generador aleatorio | `123` |
| `--block-size <n>` | Tamaño de bloque CUDA (solo V2/V3) | `256` |
| `--threads <n>` | Hilos OpenMP (solo V1) | `1` |
| `--bench-out <file>` | Archivo CSV donde se añade una fila con resultados | (vacío) |
| `-v` / `--verbose` | Modo verbose (imprime progreso) | `false` |

### Ejemplos de ejecución

```bash
# V1 — instancia pequeña, población 1024
./build/mochila_ga_cuda \
    --instance data/instance_small \
    --variant sequential \
    --pop-size 1024 \
    --generations 200 \
    --seed 42 \
    -v

# V2 — instancia mediana, población 4096, bloque 256
./build/mochila_ga_cuda \
    --instance data/instance_medium \
    --variant cuda_basic \
    --pop-size 4096 \
    --generations 200 \
    --seed 42 \
    --block-size 256 \
    -v

# V3 — instancia grande, población 16384, con salida CSV
./build/mochila_ga_cuda \
    --instance data/instance_large \
    --variant cuda_optimized \
    --pop-size 16384 \
    --generations 200 \
    --seed 42 \
    --block-size 256 \
    --bench-out results/resultados_v3.csv
```

Con Makefile:

```bash
make run instance="data/instance_small" variant="cuda_basic" seed="42"
```

---

## Experimentos (SLURM)

Los scripts para ejecutar los experimentos completos en el clúster:

| Script | Variante | Ejecuciones | Output CSV |
|--------|----------|-------------|------------|
| `experiments_v1.sh` | V1 — Secuencial | 90 principales | `results/resultados_v1.csv` |
| `experiments_v2.sh` | V2 — CUDA Básico | 90 principales + 40 bloque | `results/resultados_v2.csv`, `results/block_size_v2.csv` |
| `experiments.sh` | V3 — CUDA Optimizado | 90 principales + 40 bloque | `results/resultados_v3.csv`, `results/block_size_v3.csv` |

Enviar a SLURM:

```bash
sbatch experiments_v1.sh
sbatch experiments_v2.sh
sbatch experiments_v3.sh
```

Los logs se escriben como `logs.bench.v1.<jobid>`, `logs.bench.v2.<jobid>`, `logs.bench.<jobid>`.

### Estructura de cada experimento

- **Experimento principal**: 3 instancias × 3 poblaciones × 10 semillas = 90 corridas.  
  Instancias: `instance_small` (100 ítems), `instance_medium` (1000), `instance_large` (10000).  
  Poblaciones: 1024, 4096, 16384.  
  Semillas: 42–51.  
  Tamaño de bloque fijo: 256.

- **Experimento de bloque** (solo V2/V3): 4 tamaños × 10 semillas = 40 corridas.  
  Instancia: `instance_medium`. Población: 4096.  
  Tamaños de bloque: 64, 128, 256, 512.

- **Término**: 200 generaciones máximas o 50 generaciones sin mejora (estancamiento).

---

## Formato de salida CSV

### V1 (secuencial)

```
variant,instance,pop_size,n_items,seed,block_size,gen_exec,time_ms,
best_fitness,best_value,has_feasible,best_feasible_fitness,best_feasible_value
```

### V2 / V3 (CUDA)

```
variant,instance,pop_size,n_items,seed,block_size,gen_exec,time_ms,
kernel_time_ms,transfer_d2h_ms,transfer_h2d_ms,
best_fitness,best_value,has_feasible,best_feasible_fitness,best_feasible_value
```

Columnas extra en V2/V3:
- `kernel_time_ms`: tiempo acumulado de todos los kernels CUDA.
- `transfer_d2h_ms`: tiempo acumulado de transferencias device → host.
- `transfer_h2d_ms`: tiempo acumulado de transferencias host → device.

En V1 estas columnas no existen porque no hay kernels ni transferencias.

---

## Instancias

Cada instancia es un directorio con cuatro archivos CSV:

| Archivo | Descripción |
|---------|-------------|
| `items.csv` | id, valor, peso, volumen, categoría |
| `category_rules.csv` | categoría, mínimo, máximo |
| `incompatibilities.csv` | pares de ítems incompatibles |
| `dependencies.csv` | relaciones de dependencia (ítem, ítem requerido) |

Las capacidades máximas se fijan al 40% del total (`W = 0.4 × Σw_i`, `V = 0.4 × Σu_i`).

---

## Makefile targets

```bash
make configure       # generar archivos de build en build/
make build           # compilar
make run             # ejecutar (args: instance, variant, threads, seed)
make watch           # recompilar al guardar cambios (requiere entr)
make lint            # clang-tidy
make format          # clang-format (aplica)
make format-check    # clang-format (verifica)
make rebuild         # clean + build
make clean           # eliminar build/
```
