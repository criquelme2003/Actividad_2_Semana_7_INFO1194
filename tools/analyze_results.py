"""
Analisis completo de resultados del experimento GA.
Genera tablas de: tasas de factibilidad, calidad (best_feasible_value),
tiempos medios, speedup y eficiencia.
"""
import csv
import statistics
import collections

INPUT = "results/resultados_carlos.csv"

rows = []
with open(INPUT) as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["instance_name"] = r["instance"].split(
            "/")[-1].replace("instance_", "")
        r["time_ms_f"] = float(r["time_ms"])
        r["bfv_f"] = float(r["best_feasible_value"]
                           ) if r["best_feasible_value"] else 0.0
        r["bff_f"] = float(r["best_feasible_fitness"]
                           ) if r["best_feasible_fitness"] else 0.0
        r["feasible_i"] = int(r["feasible"])
        r["threads_i"] = int(r["threads"])
        rows.append(r)

INSTANCES = ["small", "medium", "large"]
VARIANTS = ["sequential", "parallel", "islands"]
THREAD_CFG = [1, 2, 4, 8]


def group(inst, variant, threads):
    return [r for r in rows
            if r["instance_name"] == inst
            and r["variant"] == variant
            and r["threads_i"] == threads]


# =========================================================
# 1. TASA DE FACTIBILIDAD
# =========================================================
print("=" * 70)
print("1. TASA DE FACTIBILIDAD (% runs con solución factible)")
print("=" * 70)
header = f"{'Instancia':<10} {'Variante':<12} {'Hilos':>6} {'Factible%':>10} {'N':>4}"
print(header)
print("-" * len(header))

for inst in INSTANCES:
    for variant in VARIANTS:
        t_list = [1] if variant == "sequential" else THREAD_CFG
        for t in t_list:
            g = group(inst, variant, t)
            if not g:
                continue
            rate = 100.0 * sum(r["feasible_i"] for r in g) / len(g)
            print(f"{inst:<10} {variant:<12} {t:>6} {rate:>9.1f}% {len(g):>4}")

# =========================================================
# 2. CALIDAD DE SOLUCION (best_feasible_value)
# =========================================================
print()
print("=" * 80)
print("2. CALIDAD DE SOLUCIÓN — best_feasible_value (solo runs factibles)")
print("=" * 80)
header = f"{'Instancia':<10} {'Variante':<12} {'Hilos':>6} {'Media':>12} {'StdDev':>10} {'Max':>12} {'N_fact':>7}"
print(header)
print("-" * len(header))

for inst in INSTANCES:
    for variant in VARIANTS:
        t_list = [1] if variant == "sequential" else THREAD_CFG
        for t in t_list:
            g = group(inst, variant, t)
            if not g:
                continue
            feas = [r["bfv_f"] for r in g if r["feasible_i"] == 1]
            if not feas:
                print(
                    f"{inst:<10} {variant:<12} {t:>6} {'—':>12} {'—':>10} {'—':>12} {0:>7}")
                continue
            mean = statistics.mean(feas)
            std = statistics.stdev(feas) if len(feas) > 1 else 0.0
            mx = max(feas)
            print(
                f"{inst:<10} {variant:<12} {t:>6} {mean:>12.2f} {std:>10.2f} {mx:>12.2f} {len(feas):>7}")

# =========================================================
# 3. TIEMPOS MEDIOS (ms)
# =========================================================
print()
print("=" * 80)
print("3. TIEMPOS DE EJECUCIÓN — tiempo promedio (ms)")
print("=" * 80)
header = f"{'Instancia':<10} {'Variante':<12} {'Hilos':>6} {'Media_ms':>12} {'StdDev':>10} {'Min':>10} {'Max':>10}"
print(header)
print("-" * len(header))

time_baseline = {}  # (inst, variant=sequential, t=1) -> mean_ms

for inst in INSTANCES:
    for variant in VARIANTS:
        t_list = [1] if variant == "sequential" else THREAD_CFG
        for t in t_list:
            g = group(inst, variant, t)
            if not g:
                continue
            times = [r["time_ms_f"] for r in g]
            mean = statistics.mean(times)
            std = statistics.stdev(times) if len(times) > 1 else 0.0
            mn = min(times)
            mx = max(times)
            if variant == "sequential" and t == 1:
                time_baseline[inst] = mean
            print(
                f"{inst:<10} {variant:<12} {t:>6} {mean:>12.1f} {std:>10.1f} {mn:>10.1f} {mx:>10.1f}")

# =========================================================
# 4. SPEEDUP  S_p = T1_seq / T_p
#    Usando T1_sequential como baseline universal
# =========================================================
print()
print("=" * 70)
print("4. SPEEDUP  S_p = T1_sequential / T_p")
print("=" * 70)
header = f"{'Instancia':<10} {'Variante':<12} {'Hilos':>6} {'T_media_ms':>12} {'Speedup':>10} {'Eficiencia':>12}"
print(header)
print("-" * len(header))

for inst in INSTANCES:
    t1 = time_baseline.get(inst)
    if t1 is None:
        continue
    for variant in ["parallel", "islands"]:
        for t in THREAD_CFG:
            g = group(inst, variant, t)
            if not g:
                continue
            mean = statistics.mean(r["time_ms_f"] for r in g)
            speedup = t1 / mean if mean > 0 else 0
            eff = speedup / t
            print(
                f"{inst:<10} {variant:<12} {t:>6} {mean:>12.1f} {speedup:>10.3f} {eff:>12.3f}")

# =========================================================
# 5. COMPARACION DIRECTA sequential vs parallel vs islands
#    para p=4 (punto representativo)
# =========================================================
print()
print("=" * 90)
print("5. COMPARACIÓN POR INSTANCIA — p=4 hilos (calidad vs tiempo)")
print("=" * 90)
header = f"{'Instancia':<10} {'Variante':<12} {'Hilos':>6} {'Tiempo_ms':>12} {'BFV_media':>12} {'Speedup':>10} {'Factible%':>10}"
print(header)
print("-" * len(header))

for inst in INSTANCES:
    t1 = time_baseline.get(inst)
    for variant in VARIANTS:
        t_list = [1] if variant == "sequential" else [4]
        for t in t_list:
            g = group(inst, variant, t)
            if not g:
                continue
            mean_t = statistics.mean(r["time_ms_f"] for r in g)
            speedup = (t1 / mean_t) if (t1 and mean_t > 0) else 1.0
            feas = [r["bfv_f"] for r in g if r["feasible_i"] == 1]
            bfv_m = statistics.mean(feas) if feas else 0.0
            rate = 100.0 * len(feas) / len(g)
            print(
                f"{inst:<10} {variant:<12} {t:>6} {mean_t:>12.1f} {bfv_m:>12.2f} {speedup:>10.3f} {rate:>9.1f}%")
    print()
