"""
Limpia resultados.csv manteniendo SOLO la ULTIMA ocurrencia de cada
combinacion (instance, variant, threads, seed). Los runs del experimento
final son los mas recientes en el archivo, por lo que quedan los correctos.
"""
import csv
import collections
import sys

INPUT = "results/resultados.csv"
OUTPUT = "results/resultados_clean.csv"

rows = []
with open(INPUT) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for r in reader:
        rows.append(r)

print(f"Total filas originales: {len(rows)}")

# Para sequential/small algunos seeds tienen 5 ocurrencias -- verificar fitness
# (los de 10 items tienen best_value muy bajo, < 500)
seq_small = [r for r in rows if "instance_small" in r["instance"]
             and r["variant"] == "sequential"]
print(f"\nSequential small filas: {len(seq_small)}")
by_seed = collections.Counter(r["seed"] for r in seq_small)
max_dup = max(by_seed.values())
print(f"Max duplicados por seed: {max_dup}")

# Ejemplo seed=42
for r in seq_small:
    if r["seed"] == "42":
        print(
            f"  seed=42 best_value={r['best_value']:<12} feasible={r['feasible']}")

# Mantener ULTIMA ocurrencia por clave compuesta
seen = {}
for r in rows:
    key = (r["instance"], r["variant"], r["threads"], r["seed"])
    seen[key] = r  # sobreescribe con la mas reciente

clean = list(seen.values())
print(f"\nFilas limpias: {len(clean)}")

# Verificar que son exactamente 405
expected = 405
if len(clean) != expected:
    print(f"ADVERTENCIA: esperadas {expected} filas, obtenidas {len(clean)}")

# Guardar
with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(clean)

print(f"Guardado en {OUTPUT}")

# Resumen por (instancia, variante, hilos)
print("\n=== CONTEO POR GRUPO (limpio) ===")
counts = collections.Counter()
for r in clean:
    inst = r["instance"].split("/")[-1]
    counts[(inst, r["variant"], r["threads"])] += 1

for k in sorted(counts):
    print(f"  {k[0]:<20} {k[1]:<12} t={k[2]:>2}  -> {counts[k]}")
