#!/usr/bin/env python3
"""Generate PDF plots from experiment CSVs (average across both teams)."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

THREADS = [1, 2, 4, 8]
INSTANCES = ["small", "medium", "large"]
VARIANTS = ["parallel", "islands"]


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.mean(values) if values else None


def load_rows(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            inst = r["instance"].split("/")[-1].replace("instance_", "")
            rows.append(
                {
                    "instance": inst,
                    "variant": r["variant"],
                    "threads": int(r["threads"]),
                    "time_ms": float(r["time_ms"]),
                    "bfv": float(r["best_feasible_value"]) if r["best_feasible_value"] else 0.0,
                    "feasible": int(r["feasible"]),
                }
            )
    return rows


def index_metrics(rows: List[dict]) -> Tuple[Dict[Tuple[str, str, int], List[float]], Dict[Tuple[str, str, int], List[float]]]:
    times: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    bfvs: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)

    for r in rows:
        key = (r["instance"], r["variant"], r["threads"])
        times[key].append(r["time_ms"])
        if r["feasible"] == 1:
            bfvs[key].append(r["bfv"])

    return times, bfvs


def build_speedup(times: Dict[Tuple[str, str, int], List[float]]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for inst in INSTANCES:
        t1 = _mean(times[(inst, "sequential", 1)])
        if t1 is None:
            raise ValueError(f"Missing sequential baseline for {inst}")
        inst_data: Dict[str, List[float]] = {}
        for variant in VARIANTS:
            values: List[float] = []
            for t in THREADS:
                mt = _mean(times[(inst, variant, t)])
                if mt is None:
                    raise ValueError(f"Missing time for {inst} {variant} t={t}")
                values.append(t1 / mt)
            inst_data[variant] = values
        out[inst] = inst_data
    return out


def build_quality(bfvs: Dict[Tuple[str, str, int], List[float]]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for inst in INSTANCES:
        inst_data: Dict[str, List[float]] = {}
        for variant in VARIANTS:
            values: List[float] = []
            for t in THREADS:
                mv = _mean(bfvs[(inst, variant, t)])
                if mv is None:
                    raise ValueError(f"Missing BFV for {inst} {variant} t={t}")
                values.append(mv)
            inst_data[variant] = values
        seq = _mean(bfvs[(inst, "sequential", 1)])
        if seq is None:
            raise ValueError(f"Missing sequential BFV for {inst}")
        inst_data["sequential"] = [seq]
        out[inst] = inst_data
    return out


def build_times(times: Dict[Tuple[str, str, int], List[float]]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for inst in INSTANCES:
        inst_data: Dict[str, List[float]] = {}
        for variant in VARIANTS:
            values: List[float] = []
            for t in THREADS:
                mt = _mean(times[(inst, variant, t)])
                if mt is None:
                    raise ValueError(f"Missing time for {inst} {variant} t={t}")
                values.append(mt)
            inst_data[variant] = values
        seq = _mean(times[(inst, "sequential", 1)])
        if seq is None:
            raise ValueError(f"Missing sequential time for {inst}")
        inst_data["sequential"] = [seq]
        out[inst] = inst_data
    return out


def plot_speedup(speedup: Dict[str, Dict[str, List[float]]], out_path: str) -> None:
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)

    colors = {"parallel": "#1f77b4", "islands": "#ff7f0e"}
    markers = {"parallel": "o", "islands": "s"}

    for ax, inst in zip(axes, INSTANCES):
        for variant in VARIANTS:
            ax.plot(
                THREADS,
                speedup[inst][variant],
                marker=markers[variant],
                color=colors[variant],
                linewidth=1.6,
                label=variant,
            )
        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_title(inst)
        ax.set_xlabel("Threads")
        ax.set_xticks(THREADS)

    axes[0].set_ylabel("Speedup")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    base, _ = os.path.splitext(out_path)
    fig.savefig(f"{base}.png", format="png", dpi=200)
    plt.close(fig)


def plot_quality(quality: Dict[str, Dict[str, List[float]]], out_path: str) -> None:
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)

    colors = {"parallel": "#1f77b4", "islands": "#ff7f0e", "sequential": "#333333"}
    markers = {"parallel": "o", "islands": "s", "sequential": "^"}

    for idx, (ax, inst) in enumerate(zip(axes, INSTANCES)):
        for variant in VARIANTS:
            ax.plot(
                THREADS,
                quality[inst][variant],
                marker=markers[variant],
                color=colors[variant],
                linewidth=1.6,
                label=variant,
            )
        # Sequential only at 1 thread
        ax.scatter(
            [1],
            quality[inst]["sequential"],
            marker=markers["sequential"],
            color=colors["sequential"],
            label="sequential" if idx == 0 else "_nolegend_",
            zorder=3,
        )
        ax.set_title(inst)
        ax.set_xlabel("Threads")
        ax.set_xticks(THREADS)

    axes[0].set_ylabel("Best feasible value (mean)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    base, _ = os.path.splitext(out_path)
    fig.savefig(f"{base}.png", format="png", dpi=200)
    plt.close(fig)


def plot_times(times: Dict[str, Dict[str, List[float]]], out_path: str) -> None:
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)

    colors = {"parallel": "#1f77b4", "islands": "#ff7f0e", "sequential": "#333333"}
    markers = {"parallel": "o", "islands": "s", "sequential": "^"}

    for idx, (ax, inst) in enumerate(zip(axes, INSTANCES)):
        for variant in VARIANTS:
            ax.plot(
                THREADS,
                times[inst][variant],
                marker=markers[variant],
                color=colors[variant],
                linewidth=1.6,
                label=variant,
            )
        ax.scatter(
            [1],
            times[inst]["sequential"],
            marker=markers["sequential"],
            color=colors["sequential"],
            label="sequential" if idx == 0 else "_nolegend_",
            zorder=3,
        )
        ax.set_title(inst)
        ax.set_xlabel("Threads")
        ax.set_xticks(THREADS)

    axes[0].set_ylabel("Time (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    base, _ = os.path.splitext(out_path)
    fig.savefig(f"{base}.png", format="png", dpi=200)
    plt.close(fig)


def plot_time_ratio(time_ratio: Dict[str, Dict[str, List[float]]], out_path: str) -> None:
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)

    colors = {"parallel": "#1f77b4", "islands": "#ff7f0e", "sequential": "#333333"}
    markers = {"parallel": "o", "islands": "s", "sequential": "^"}

    for idx, (ax, inst) in enumerate(zip(axes, INSTANCES)):
        for variant in VARIANTS:
            ax.plot(
                THREADS,
                time_ratio[inst][variant],
                marker=markers[variant],
                color=colors[variant],
                linewidth=1.6,
                label=variant,
            )
        ax.scatter(
            [1],
            time_ratio[inst]["sequential"],
            marker=markers["sequential"],
            color=colors["sequential"],
            label="sequential" if idx == 0 else "_nolegend_",
            zorder=3,
        )
        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_title(inst)
        ax.set_xlabel("Threads")
        ax.set_xticks(THREADS)

    axes[0].set_ylabel("Carlos / Benja (time ratio)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    base, _ = os.path.splitext(out_path)
    fig.savefig(f"{base}.png", format="png", dpi=200)
    plt.close(fig)


def _generate_set(rows: List[dict], out_dir: str, tag: str) -> List[str]:
    times, bfvs = index_metrics(rows)
    speedup = build_speedup(times)
    quality = build_quality(bfvs)
    times_series = build_times(times)

    speedup_path = os.path.join(out_dir, f"speedup_{tag}.png")
    quality_path = os.path.join(out_dir, f"quality_{tag}.png")
    time_path = os.path.join(out_dir, f"time_{tag}.png")

    plot_speedup(speedup, speedup_path)
    plot_quality(quality, quality_path)
    plot_times(times_series, time_path)

    return [speedup_path, quality_path, time_path]


def _mean_time_table(rows: List[dict]) -> Dict[Tuple[str, str, int], float]:
    times, _ = index_metrics(rows)
    table: Dict[Tuple[str, str, int], float] = {}
    for inst in INSTANCES:
        for variant in ["sequential"] + VARIANTS:
            t_list = [1] if variant == "sequential" else THREADS
            for t in t_list:
                table[(inst, variant, t)] = _mean(times[(inst, variant, t)])
    return table


def build_time_ratio(rows_benja: List[dict], rows_carlos: List[dict]) -> Dict[str, Dict[str, List[float]]]:
    benja = _mean_time_table(rows_benja)
    carlos = _mean_time_table(rows_carlos)

    out: Dict[str, Dict[str, List[float]]] = {}
    for inst in INSTANCES:
        inst_data: Dict[str, List[float]] = {}
        for variant in VARIANTS:
            ratios: List[float] = []
            for t in THREADS:
                cb = carlos[(inst, variant, t)] / benja[(inst, variant, t)]
                ratios.append(cb)
            inst_data[variant] = ratios
        inst_data["sequential"] = [carlos[(inst, "sequential", 1)] / benja[(inst, "sequential", 1)]]
        out[inst] = inst_data
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PNG plots from CSV results.")
    parser.add_argument("--benja", default="results/resultados_benja.csv", help="Path to Benja CSV")
    parser.add_argument("--carlos", default="results/resultados_carlos.csv", help="Path to Carlos CSV")
    parser.add_argument("--out-dir", default="results", help="Output directory for plots")
    parser.add_argument("--avg", action="store_true", help="Also generate averaged plots")
    args = parser.parse_args()

    rows_benja = load_rows(args.benja)
    rows_carlos = load_rows(args.carlos)

    os.makedirs(args.out_dir, exist_ok=True)
    # Clean previous PNG outputs to avoid stale plots.
    for filename in os.listdir(args.out_dir):
        if filename.endswith(".png"):
            os.remove(os.path.join(args.out_dir, filename))

    generated: List[str] = []
    generated += _generate_set(rows_benja, args.out_dir, "benja")
    generated += _generate_set(rows_carlos, args.out_dir, "carlos")

    ratio = build_time_ratio(rows_benja, rows_carlos)
    ratio_path = os.path.join(args.out_dir, "time_ratio.png")
    plot_time_ratio(ratio, ratio_path)
    generated.append(ratio_path)

    if args.avg:
        rows_avg = rows_benja + rows_carlos
        generated += _generate_set(rows_avg, args.out_dir, "avg")

    print("Generated:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
