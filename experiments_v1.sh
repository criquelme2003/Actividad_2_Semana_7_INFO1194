#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=bench_cuda_v1
#SBATCH --partition=RunCuda
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=logs.bench.v1.%j
#SBATCH --time=04:00:00

# ============================================================
# Toma de tiempos — Variante 1: Secuencial
# Instancias: small (100 ítems), medium (1000), large (10000)
# Poblaciones: 1024, 4096, 16384
# Semillas: 42–51 (10 repeticiones por configuración)
# ============================================================

set -e

PROJ=/home/grupo1/Actividad_2_Semana_7_INFO1194
BINARY=$PROJ/build/mochila_ga_cuda
RESULTS=$PROJ/results
MAIN_CSV=$RESULTS/resultados_v1.csv
BLOCK_CSV=$RESULTS/block_size_v1.csv

mkdir -p "$RESULTS"

cd "$PROJ/build"
cmake --build . -j8 --target mochila_ga_cuda 2>&1 | tail -3
cd "$PROJ"

echo "Hardware:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

HEADER="variant,instance,pop_size,n_items,seed,block_size,gen_exec,time_ms,best_fitness,best_value,has_feasible,best_feasible_fitness,best_feasible_value"
echo "$HEADER" > "$MAIN_CSV"
echo "$HEADER" > "$BLOCK_CSV"

INSTANCES="instance_small instance_medium instance_large"
POP_SIZES="1024 4096 16384"
SEEDS="42 43 44 45 46 47 48 49 50 51"
GENS=200
DEFAULT_BLOCK=256
VAR=sequential

echo "=== EXPERIMENTO PRINCIPAL: $VAR ==="
n=0
for inst in $INSTANCES; do
    for pop in $POP_SIZES; do
        for seed in $SEEDS; do n=$((n+1)); done
    done
done
echo "Total: $n ejecuciones"

done=0
for inst in $INSTANCES; do
    for pop in $POP_SIZES; do
        for seed in $SEEDS; do
            done=$((done+1))
            echo "[$done/$n] $inst | pop=$pop | seed=$seed"
            $BINARY \
                --instance "$PROJ/data/$inst" \
                --variant  "$VAR" \
                --pop-size "$pop" \
                --generations "$GENS" \
                --seed "$seed" \
                --block-size "$DEFAULT_BLOCK" \
                --bench-out "$MAIN_CSV" 2>/dev/null \
            || echo "  WARN: falló $inst pop=$pop seed=$seed"
        done
    done
done

echo ""
echo "=== EFECTO DEL TAMAÑO DE BLOQUE ==="
echo "No aplica para variante secuencial (block_size es un parámetro CUDA)."
echo "El CSV $BLOCK_CSV queda solo con el header."

echo ""
echo "=== LISTO ==="
echo "Principal: $MAIN_CSV  ($(wc -l < "$MAIN_CSV") filas)"
echo "Block:     $BLOCK_CSV (solo header — no aplica para secuencial)"
