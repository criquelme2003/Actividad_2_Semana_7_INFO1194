#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=bench_v3_timing
#SBATCH --partition=RunCuda
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=logs.bench.v3.%j
#SBATCH --time=02:00:00

# ============================================================
# Toma de tiempos V3 (cuda_optimized) con desglose kernel/transfer
# CSVs nuevos: resultados_v3_timing.csv, block_size_v3_timing.csv
# ============================================================

set -e

PROJ=/home/grupo1/Actividad_2_Semana_7_INFO1194
BINARY=$PROJ/build/mochila_ga_cuda
RESULTS=$PROJ/results
MAIN_CSV=$RESULTS/resultados_v3_timing.csv
BLOCK_CSV=$RESULTS/block_size_v3_timing.csv

mkdir -p "$RESULTS"

cd "$PROJ/build"
cmake --build . -j8 --target mochila_ga_cuda 2>&1 | tail -3
cd "$PROJ"

echo "Hardware:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

HEADER="variant,instance,pop_size,n_items,seed,block_size,gen_exec,time_ms,kernel_time_ms,transfer_d2h_ms,transfer_h2d_ms,best_fitness,best_value,has_feasible,best_feasible_fitness,best_feasible_value"
echo "$HEADER" > "$MAIN_CSV"
echo "$HEADER" > "$BLOCK_CSV"

INSTANCES="instance_small instance_medium instance_large"
POP_SIZES="1024 4096 16384"
SEEDS="42 43 44 45 46 47 48 49 50 51"
GENS=200
DEFAULT_BLOCK=256
VAR=cuda_optimized

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
BLOCK_SIZES="64 128 256 512"
BLOCK_INST="instance_medium"
BLOCK_POP=4096

nb=0
for blk in $BLOCK_SIZES; do for seed in $SEEDS; do nb=$((nb+1)); done; done
echo "Total: $nb ejecuciones"

db=0
for blk in $BLOCK_SIZES; do
    for seed in $SEEDS; do
        db=$((db+1))
        echo "[$db/$nb] block=$blk | seed=$seed"
        $BINARY \
            --instance "$PROJ/data/$BLOCK_INST" \
            --variant  "$VAR" \
            --pop-size "$BLOCK_POP" \
            --generations "$GENS" \
            --seed "$seed" \
            --block-size "$blk" \
            --bench-out "$BLOCK_CSV" 2>/dev/null \
        || echo "  WARN: falló block=$blk seed=$seed"
    done
done

echo ""
echo "=== LISTO ==="
echo "Principal: $MAIN_CSV  ($(wc -l < "$MAIN_CSV") filas)"
echo "Block:     $BLOCK_CSV ($(wc -l < "$BLOCK_CSV") filas)"
