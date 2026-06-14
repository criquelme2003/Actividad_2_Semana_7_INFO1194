#!/usr/bin/env python3
"""
Script de experimentación para el problema de la mochila extendida.

Ejecuta todas las combinaciones:
  variante  × hilos × instancia × semilla
y acumula los resultados en results/resultados.csv.

Diseño experimental (§7 del enunciado):
  - Instancias: small (100), medium (1000), large (10000)
  - Variantes:  sequential, parallel, islands
  - Hilos:      1, 2, 4, 8  (sequential siempre con 1 hilo)
  - Semillas:   15 semillas fijas para reproducibilidad

Métricas registradas por el binario (§8):
  instance, variant, threads, seed, generations,
  best_fitness, best_value,
  best_feasible_fitness, best_feasible_value,
  feasible, time_ms,
  best_feasible_items, best_feasible_weight, best_feasible_volume

Speed-up y eficiencia se calculan en post-proceso a partir de time_ms.

Uso (desde WSL o Linux, en la raíz del proyecto):
  python3 tools/run_experiments.py
  python3 tools/run_experiments.py --dry-run        # imprime comandos sin ejecutar
  python3 tools/run_experiments.py --only-instance small
"""

import argparse
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Configuración experimental
# ---------------------------------------------------------------------------

BINARY = "./build/mochila_ga.exe" if sys.platform == "win32" else "./build/mochila_ga"

INSTANCES = {
    "small":  "data/instance_small",
    "medium": "data/instance_medium",
    "large":  "data/instance_large",
}

VARIANTS = ["sequential", "parallel", "islands"]

# Hilos a probar por variante. Sequential no tiene paralelismo interno:
# se mide una sola vez con 1 hilo para obtener T1 (línea base de speed-up).
THREADS_BY_VARIANT: dict[str, list[int]] = {
    "sequential": [1],
    "parallel":   [1, 2, 4, 8],
    "islands":    [1, 2, 4, 8],
}

# 15 semillas registradas (reproducibilidad §7).
SEEDS = [42, 123, 256, 512, 999, 1337, 2048, 3141, 4096, 5000,
         6789, 7654, 8192, 9001, 9999]

OUTPUT_CSV = "results/resultados.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _total_runs(only_instance: str | None) -> int:
    instances = {
        only_instance: INSTANCES[only_instance]} if only_instance else INSTANCES
    total = 0
    for _ in instances:
        for variant in VARIANTS:
            for threads in THREADS_BY_VARIANT[variant]:
                total += len(SEEDS)
    return total


def _fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def run_one(instance_path: str, variant: str, threads: int, seed: int,
            dry_run: bool) -> tuple[bool, float]:
    """Ejecuta el binario y devuelve (éxito, tiempo_real_segundos)."""
    cmd = [
        BINARY,
        "--instance", instance_path,
        "--variant",  variant,
        "--threads",  str(threads),
        "--seed",     str(seed),
        "--output-csv", OUTPUT_CSV,
    ]

    if dry_run:
        print(" ".join(cmd))
        return True, 0.0

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        # Imprime stderr para facilitar diagnóstico sin detener el experimento.
        print(
            f"  [WARN] stderr: {result.stderr.strip()[:120]}", file=sys.stderr)

    return result.returncode == 0, elapsed


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta todos los experimentos y guarda resultados en CSV."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Imprime los comandos sin ejecutarlos.")
    parser.add_argument("--only-instance", choices=list(INSTANCES.keys()),
                        help="Ejecuta solo la instancia indicada.")
    args = parser.parse_args()

    if not args.dry_run and not os.path.isfile(BINARY):
        print(f"Error: binario '{BINARY}' no encontrado. Compila primero con 'make build'.",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    instances = ({args.only_instance: INSTANCES[args.only_instance]}
                 if args.only_instance else INSTANCES)

    total = _total_runs(args.only_instance)
    done = 0
    failed = 0
    t_start = time.monotonic()

    print(f"Experimentos totales: {total}")
    print(f"Semillas: {SEEDS}")
    print(f"Salida: {OUTPUT_CSV}\n")

    for inst_name, inst_path in instances.items():
        print(f"=== Instancia: {inst_name} ({inst_path}) ===")
        for variant in VARIANTS:
            for threads in THREADS_BY_VARIANT[variant]:
                for seed in SEEDS:
                    ok, run_secs = run_one(inst_path, variant, threads, seed,
                                           args.dry_run)
                    done += 1
                    if not ok:
                        failed += 1

                    elapsed_total = time.monotonic() - t_start
                    avg_per_run = elapsed_total / done if done else 0
                    eta = avg_per_run * (total - done)

                    status = "OK  " if ok else "FAIL"
                    print(
                        f"  [{done:4d}/{total}] {status} "
                        f"{inst_name:<8} {variant:<12} t={threads} seed={seed:5d} "
                        f"({run_secs:.1f}s)  ETA {_fmt_elapsed(eta)}"
                    )

        print()

    elapsed_total = time.monotonic() - t_start
    print(f"Completado en {_fmt_elapsed(elapsed_total)}. "
          f"Exitosos: {done - failed}/{total}. "
          f"Fallidos: {failed}.")
    if not args.dry_run:
        print(f"Resultados en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
