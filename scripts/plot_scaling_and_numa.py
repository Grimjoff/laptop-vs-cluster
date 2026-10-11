"""
Plotting Pipeline for Strong Scaling & cc-NUMA Analysis (Day 10 — HPC SoSe 2026)
================================================================================
Generates the mandatory publication-grade figures (300 DPI) for the portfolio report:
- Graphic 2: Strong Scaling Speedup (S(p) vs. Ideal S=p)
- Graphic 3: Parallel Efficiency & cc-NUMA Analysis (E(p) in % with 12-core socket boundary)
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_strong_scaling_and_numa(
    csv_path: str = "results/benchmark_strong_scaling.csv",
    output_dir: str = "figures",
    show_plots: bool = True,
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    df = pd.read_csv(csv_path).sort_values("cores")
    os.makedirs(output_dir, exist_ok=True)

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    cores = df["cores"].to_numpy()
    durations = df["duration_sec"].to_numpy()
    
    # Calculate baseline and speedup if not present
    t1 = durations[0]
    speedups = df["speedup"].to_numpy() if "speedup" in df.columns else t1 / durations
    efficiencies = (df["efficiency"].to_numpy() if "efficiency" in df.columns else speedups / cores) * 100.0

    # --------------------------------------------------------------------------
    # Grafik 2: Strong Scaling Speedup (Empirisch vs. Ideal S = p)
    # --------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    # Ideal Speedup Line (S = p)
    max_cores = cores[-1]
    ideal_x = np.array([1, max_cores])
    ideal_y = np.array([1, max_cores])
    ax1.plot(
        ideal_x,
        ideal_y,
        linestyle="--",
        color="#dc2626",
        linewidth=2.0,
        label=r"Idealer linearer Speedup ($S = p$)",
        zorder=2,
    )

    # Empirical Speedup Line
    ax1.plot(
        cores,
        speedups,
        marker="o",
        markersize=8,
        color="#2563eb",
        linewidth=2.4,
        label=r"Empirischer Speedup ($S = T_1 / T_p$)",
        zorder=3,
    )

    # Annotate empirical values
    for x, y, dur in zip(cores, speedups, durations):
        ax1.annotate(
            f"{y:.2f}x\n({dur:.1f}s)",
            xy=(x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#1e293b",
        )

    ax1.set_title("Grafik 2: Strong Scaling Speedup auf dem HPC-Cluster", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xlabel("Anzahl CPU-Kerne (p)", fontsize=11, labelpad=10)
    ax1.set_ylabel("Speedup-Faktor S(p)", fontsize=11, labelpad=10)
    ax1.set_xticks(cores)
    ax1.set_xlim(0, max_cores + 1)
    ax1.set_ylim(0, max_cores * 1.15)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper left")

    fig1.tight_layout()
    path_graphic2 = os.path.join(output_dir, "strong_scaling_speedup.png")
    fig1.savefig(path_graphic2, dpi=300)
    print(f"Grafik 2 erfolgreich gespeichert: {path_graphic2}")

    # --------------------------------------------------------------------------
    # Grafik 3: Parallele Effizienz & cc-NUMA Analyse (Sockelübergang bei 12 Kernen)
    # --------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    # 100% Efficiency baseline
    ax2.axhline(100.0, color="#94a3b8", linestyle=":", linewidth=1.5, label="100% Maximaleffizienz")

    # Empirical Efficiency Line
    ax2.plot(
        cores,
        efficiencies,
        marker="s",
        markersize=8,
        color="#059669",
        linewidth=2.4,
        label="Parallele Effizienz E(p) [%]",
        zorder=3,
    )

    # cc-NUMA Sockelgrenze (12 Kerne) hervorheben
    if 12 in cores or max_cores >= 12:
        ax2.axvline(
            12,
            color="#d97706",
            linestyle="-.",
            linewidth=2.2,
            label="cc-NUMA Grenze (Sockel 1: 12 Kerne)",
            zorder=2,
        )
        # Highlight regions
        ax2.axvspan(0, 12, alpha=0.06, color="#3b82f6", label="Sockel 1 (Lokaler Speicher)")
        if max_cores > 12:
            ax2.axvspan(12, max_cores + 1, alpha=0.06, color="#ef4444", label="Sockel 2 (UPI-Bus Interconnect)")

    # Annotate efficiency percentages
    for x, y in zip(cores, efficiencies):
        ax2.annotate(
            f"{y:.1f}%",
            xy=(x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#064e3b",
        )

    ax2.set_title("Grafik 3: Parallele Effizienz & cc-NUMA Analyse (Intel Xeon)", fontsize=13, fontweight="bold", pad=15)
    ax2.set_xlabel("Anzahl CPU-Kerne (p)", fontsize=11, labelpad=10)
    ax2.set_ylabel("Parallele Effizienz E(p) in %", fontsize=11, labelpad=10)
    ax2.set_xticks(cores)
    ax2.set_xlim(0, max_cores + 1)
    ax2.set_ylim(0, 115)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="lower left", fontsize=8.5)

    fig2.tight_layout()
    path_graphic3 = os.path.join(output_dir, "parallel_efficiency_numa.png")
    fig2.savefig(path_graphic3, dpi=300)
    print(f"Grafik 3 erfolgreich gespeichert: {path_graphic3}")

    if show_plots:
        plt.show()

    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Strong Scaling & cc-NUMA figures.")
    parser.add_argument("--csv", default="results/benchmark_strong_scaling.csv", help="Input benchmark CSV file")
    parser.add_argument("--output-dir", default="figures", help="Directory to save figures")
    parser.add_argument("--no-show", action="store_true", help="Do not open GUI windows")
    args = parser.parse_args()

    plot_strong_scaling_and_numa(
        csv_path=args.csv,
        output_dir=args.output_dir,
        show_plots=not args.no_show,
    )
