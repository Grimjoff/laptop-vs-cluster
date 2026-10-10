import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_benchmark_medians(csv_path="benchmark_results.csv", output_path="figures/benchmark_median_runtime.png", show_plot=True):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # 1. Load benchmark results
    df = pd.read_csv(csv_path)

    # 2. Compute median duration per n_jobs setup
    summary = (
        df.groupby("n_jobs")["duration_sec"]
        .agg(median="median", min="min", max="max", count="count")
        .reset_index()
        .sort_values("n_jobs")
    )

    # 3. Calculate ideal scaling (halves whenever cores double: T_ideal = T_1 / p)
    t1 = summary.loc[summary["n_jobs"] == 1, "median"].values[0]
    summary["ideal"] = t1 / summary["n_jobs"]

    print("--- Benchmark Summary (Median of Runs) ---")
    print(summary.to_string(index=False))

    # 4. Create plot
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    # Optional: Plot individual runs as light dots for transparency
    ax.scatter(
        df["n_jobs"],
        df["duration_sec"],
        color="#94a3b8",
        alpha=0.6,
        s=35,
        label="Individual Runs",
        zorder=2,
    )

    # Plot actual median duration
    ax.plot(
        summary["n_jobs"],
        summary["median"],
        marker="o",
        markersize=8,
        linewidth=2.2,
        color="#2563eb",
        label="Median Duration (Empirical)",
        zorder=3,
    )

    # Plot ideal scaling line (starts at n_jobs=1 and halves as cores double)
    ax.plot(
        summary["n_jobs"],
        summary["ideal"],
        linestyle="--",
        marker="s",
        markersize=6,
        linewidth=2.0,
        color="#dc2626",
        label="Ideal Scaling ($T_1 / p$, Halving)",
        zorder=3,
    )

    # Annotate empirical median values above points
    for _, row in summary.iterrows():
        n_jobs = int(row["n_jobs"])
        med_val = row["median"]
        ax.annotate(
            f"{med_val:.1f}s",
            xy=(n_jobs, med_val),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1e293b",
        )

    # Annotate ideal scaling values below points (skip n_jobs=1 to prevent label overlap)
    for _, row in summary.iterrows():
        n_jobs = int(row["n_jobs"])
        if n_jobs == 1:
            continue
        ideal_val = row["ideal"]
        ax.annotate(
            f"{ideal_val:.1f}s",
            xy=(n_jobs, ideal_val),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            fontstyle="italic",
            color="#991b1b",
        )

    # Configure axes, labels, and title
    ax.set_title("Benchmark Runtime vs. CPU Cores / Jobs (Empirical vs. Ideal Scaling)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Jobs / Cores (n_jobs)", fontsize=11, labelpad=10)
    ax.set_ylabel("Duration (seconds)", fontsize=11, labelpad=10)
    ax.set_xticks(summary["n_jobs"].tolist())
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")

    # Set reasonable y-limit margins for annotation headroom
    y_max = summary["median"].max() * 1.08
    ax.set_ylim(bottom=0, top=y_max)

    fig.tight_layout()

    # 4. Save and optionally show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
        print(f"\nPlot saved to: {output_path}")

    if show_plot:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot median benchmark duration vs n_jobs.")
    parser.add_argument("--csv", default="benchmark_results.csv", help="Path to benchmark CSV file")
    parser.add_argument("--output", default="figures/benchmark_median_runtime.png", help="Path to save output plot")
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive plot window")
    args = parser.parse_args()

    plot_benchmark_medians(
        csv_path=args.csv,
        output_path=args.output,
        show_plot=not args.no_show,
    )
