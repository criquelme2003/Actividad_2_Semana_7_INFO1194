#!/usr/bin/env python3
"""
Generador de instancias para el problema de la mochila extendida.

Produce cuatro archivos CSV dentro de un directorio de salida:
  items.csv            — id, valor, peso, volumen, categoria
  category_rules.csv   — categoria, minimo, maximo
  incompatibilities.csv— id_item_a, id_item_b
  dependencies.csv     — id_item, id_requerido

Uso:
  python3 tools/generate_instance.py --items 1000 --output data/instance_medium --seed 42
  python3 tools/generate_instance.py --items 10000 --output data/instance_large  --seed 42
"""

import argparse
import csv
import os
import random

CATEGORIES = ["tools", "food", "electronics", "medical", "clothing"]

# Límites de número de pares de restricciones escalados para que la evaluación
# de fitness no domine el tiempo de ejecución en instancias grandes.
# El enunciado no fija cantidad; estos valores producen instancias razonablemente
# difíciles sin ser irresolubles.
_INCOMPAT_SCALE = [(100, 50), (1_000, 200), (10_000, 500)]
_DEPS_SCALE = [(100, 20), (1_000, 100), (10_000, 300)]


def _target_count(n: int, scale_table: list) -> int:
    """Interpola linealmente el número objetivo de pares según n."""
    for threshold, count in sorted(scale_table, reverse=True):
        if n >= threshold:
            return count
    return scale_table[0][1]


def generate_instance(n_items: int, output_dir: str, seed: int) -> None:
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # items.csv
    # Valores en [5, 100], pesos en [1, 50], volúmenes en [1, 40].
    # Rango elegido para que W = 0.40 × suma_pesos produzca una instancia
    # no trivial: ni vacía ni casi llena.
    # ------------------------------------------------------------------
    items = [
        {
            "id":        i,
            "valor":     round(rng.uniform(5.0, 100.0), 2),
            "peso":      round(rng.uniform(1.0, 50.0),  2),
            "volumen":   round(rng.uniform(1.0, 40.0),  2),
            "categoria": rng.choice(CATEGORIES),
        }
        for i in range(1, n_items + 1)
    ]

    with open(os.path.join(output_dir, "items.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "valor", "peso", "volumen", "categoria"])
        writer.writeheader()
        writer.writerows(items)

    # ------------------------------------------------------------------
    # category_rules.csv
    # Mínimo = 0 (restricción blanda según §5.1).
    # Máximo = 40 % de los ítems de esa categoría, al menos 1.
    # Esto limita las soluciones sin hacerlas inviables.
    # ------------------------------------------------------------------
    cat_counts: dict[str, int] = {}
    for item in items:
        cat_counts[item["categoria"]] = cat_counts.get(
            item["categoria"], 0) + 1

    rules = []
    for cat in CATEGORIES:
        count = cat_counts.get(cat, 0)
        max_val = max(1, int(count * 0.40)) if count > 0 else 1
        rules.append({"categoria": cat, "minimo": 0, "maximo": max_val})

    with open(os.path.join(output_dir, "category_rules.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["categoria", "minimo", "maximo"])
        writer.writeheader()
        writer.writerows(rules)

    # ------------------------------------------------------------------
    # incompatibilities.csv  (restricción dura: §5.1)
    # Pares sin orden (a < b) para evitar duplicados.
    # Cantidad acotada para que el loop de evaluación en instancias grandes
    # no domine el tiempo secuencial.
    # ------------------------------------------------------------------
    ids = [item["id"] for item in items]
    n_incompat = _target_count(n_items, _INCOMPAT_SCALE)
    incompat_set: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = n_incompat * 20

    while len(incompat_set) < n_incompat and attempts < max_attempts:
        a, b = rng.sample(ids, 2)
        if a > b:
            a, b = b, a
        incompat_set.add((a, b))
        attempts += 1

    with open(os.path.join(output_dir, "incompatibilities.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id_item_a", "id_item_b"])
        writer.writeheader()
        for a, b in sorted(incompat_set):
            writer.writerow({"id_item_a": a, "id_item_b": b})

    # ------------------------------------------------------------------
    # dependencies.csv  (restricción dura: §5.1)
    # Formato: si se selecciona id_item, debe seleccionarse id_requerido.
    # Para evitar ciclos: id_requerido < id_item siempre.
    # ------------------------------------------------------------------
    n_deps = _target_count(n_items, _DEPS_SCALE)
    dep_set: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = n_deps * 20

    while len(dep_set) < n_deps and attempts < max_attempts:
        a, b = rng.sample(ids, 2)
        # Garantizar id_requerido < id_item para evitar ciclos directos
        item_id, req_id = (max(a, b), min(a, b))
        dep_set.add((item_id, req_id))
        attempts += 1

    with open(os.path.join(output_dir, "dependencies.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id_item", "id_requerido"])
        writer.writeheader()
        for item_id, req_id in sorted(dep_set):
            writer.writerow({"id_item": item_id, "id_requerido": req_id})

    # Resumen para verificación
    total_weight = sum(item["peso"] for item in items)
    total_volume = sum(item["volumen"] for item in items)
    print(f"Instancia generada en '{output_dir}':")
    print(f"  Items:             {n_items}")
    print(
        f"  Suma pesos:        {total_weight:.1f}  →  W (40%) = {total_weight * 0.40:.1f}")
    print(
        f"  Suma volúmenes:    {total_volume:.1f}  →  V (40%) = {total_volume * 0.40:.1f}")
    print(f"  Incompatibilidades:{len(incompat_set)}")
    print(f"  Dependencias:      {len(dep_set)}")
    for cat in CATEGORIES:
        print(f"  Categoría {cat}: {cat_counts.get(cat, 0)} ítems")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generador de instancias para el problema de la mochila extendida."
    )
    parser.add_argument("--items",  type=int, required=True,
                        help="Número de ítems a generar (e.g. 100, 1000, 10000)")
    parser.add_argument("--output", type=str, required=True,
                        help="Directorio de salida (se crea si no existe)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Semilla para el generador (default: 42)")
    args = parser.parse_args()

    generate_instance(args.items, args.output, args.seed)


if __name__ == "__main__":
    main()
