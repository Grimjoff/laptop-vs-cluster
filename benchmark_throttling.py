import os
import sys
import time
import argparse
import platform
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Optional xgboost support if installed
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def run_throttling_benchmark(
    parquet_path="data/delays_zurich_transport_45034.parquet",
    csv_file="throttling_results.csv",
    plot_path="figures/thermal_throttling_runs.png",
    total_runs=15,
    model_type="rf",
    sample_size=None,
    n_jobs=-1,
    machine_label=None,
    n_estimators=100,
    max_depth=15,
    random_state=42,
    show_plot=False,
):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

    # Detect machine info if not provided
    if machine_label is None:
        machine_label = f"{platform.system()}_{platform.machine()}"

    max_cores = os.cpu_count() or 1
    active_cores = max_cores if (n_jobs == -1 or n_jobs is None) else min(n_jobs, max_cores)

    print("=" * 70)
    print("        HPC THERMAL THROTTLING BENCHMARK (Continuous Load)        ")
    print("=" * 70)
    print(f"Machine Label      : {machine_label}")
    print(f"Platform           : {platform.platform()}")
    print(f"Detected CPU Cores : {max_cores} physical/logical")
    print(f"Active Worker Cores: {active_cores} (n_jobs={n_jobs})")
    print(f"Model Architecture : {model_type.upper()}")
    print(f"Total Iterations   : {total_runs} consecutive runs (ZERO delay)")
    print("=" * 70)

    # 1. Load dataset
    print(f"\nLoading dataset: {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)
    print(f"Loaded {total_rows:,} rows and {len(df.columns)} columns.")

    if sample_size and sample_size < total_rows:
        print(f"Using a sample of {sample_size:,} rows for benchmark...")
        data = df.sample(n=sample_size, random_state=random_state)
    else:
        data = df

    # 2. Features and split
    target_col = "delay"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    print(f"Training set: {len(X_train):,} samples\n")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(csv_file) or ".", exist_ok=True)
    if plot_path:
        os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)

    results = []
    baseline_time = None

    print("-" * 70)
    print(f"{'Run':<6} | {'Duration (s)':<14} | {'vs Run 1':<12} | {'Timestamp':<20}")
    print("-" * 70)

    for run_idx in range(1, total_runs + 1):
        # Instantiate model for each iteration to clear cached tree allocations
        if model_type.lower() == "rf":
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=active_cores,
                random_state=random_state,
            )
        elif model_type.lower() == "xgb":
            if not HAS_XGB:
                raise ImportError("xgboost is not installed. Run `pip install xgboost` or use --model rf")
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                tree_method="hist",
                n_jobs=active_cores,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'rf' or 'xgb'.")

        timestamp_start = time.strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.time()
        model.fit(X_train, y_train)
        duration = round(time.time() - t0, 4)

        if baseline_time is None:
            baseline_time = duration
            delta_str = "Baseline"
            slowdown_pct = 0.0
        else:
            slowdown_pct = round(((duration - baseline_time) / baseline_time) * 100, 2)
            sign = "+" if slowdown_pct >= 0 else ""
            delta_str = f"{sign}{slowdown_pct:.1f}%"

        print(f"Run {run_idx:<2} | {duration:10.2f}s    | {delta_str:<12} | {timestamp_start}")

        record = {
            "machine": machine_label,
            "model": model_type.upper(),
            "n_jobs": active_cores,
            "sample_size": len(data),
            "run_index": run_idx,
            "duration_sec": duration,
            "slowdown_pct": slowdown_pct,
            "timestamp": timestamp_start,
        }
        results.append(record)

        # Append immediately to CSV so progress is saved even if stopped early
        row_df = pd.DataFrame([record])
        file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
        row_df.to_csv(csv_file, mode="a", header=not file_exists, index=False)

    df_results = pd.DataFrame(results)

    print("-" * 70)
    print(f"\nAll {total_runs} runs completed successfully.")
    print(f"Results appended to: {csv_file}")
    
    first_run = df_results.iloc[0]["duration_sec"]
    last_run = df_results.iloc[-1]["duration_sec"]
    overall_slowdown = ((last_run - first_run) / first_run) * 100
    print(f"\nSummary:")
    print(f"  Run 1  (Cold baseline) : {first_run:.2f}s")
    print(f"  Run {total_runs} (Heat-soaked)   : {last_run:.2f}s")
    print(f"  Total Thermal Slowdown : {overall_slowdown:+.1f}%\n")

    # 3. Generate Plot (Grafik 1: Thermal Throttling)
    if plot_path:
        _plot_throttling(df_results, plot_path, machine_label, show_plot)


def _plot_throttling(df_results, plot_path, machine_label, show_plot):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    runs = df_results["run_index"]
    durations = df_results["duration_sec"]

    ax.plot(
        runs,
        durations,
        marker="o",
        linewidth=2.2,
        markersize=7,
        color="#e11d48",
        label=f"{machine_label} (Sustained Load)",
        zorder=3,
    )

    # Highlight cold baseline and heat-soaked run
    r1_val = durations.iloc[0]
    r_end_val = durations.iloc[-1]
    ax.annotate(
        f"Run 1: {r1_val:.1f}s\n(Cold)",
        xy=(runs.iloc[0], r1_val),
        xytext=(15, 15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1),
        fontweight="bold",
        fontsize=9,
        color="#1e293b",
    )

    diff_pct = ((r_end_val - r1_val) / r1_val) * 100
    sign = "+" if diff_pct >= 0 else ""
    ax.annotate(
        f"Run {runs.iloc[-1]}: {r_end_val:.1f}s\n({sign}{diff_pct:.1f}%)",
        xy=(runs.iloc[-1], r_end_val),
        xytext=(-45, 15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1),
        fontweight="bold",
        fontsize=9,
        color="#9f1239",
    )

    ax.set_title("Thermal Throttling: Training Duration vs. Consecutive Iterations", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Consecutive Run Index (1 to 15)", fontsize=11, labelpad=10)
    ax.set_ylabel("Execution Time (seconds)", fontsize=11, labelpad=10)
    ax.set_xticks(runs.tolist())
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")

    # Adjust y limits for aesthetic margins
    y_min = max(0, durations.min() * 0.9)
    y_max = durations.max() * 1.15
    ax.set_ylim(bottom=y_min, top=y_max)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    print(f"Throttling plot saved to: {plot_path}")

    if show_plot:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thermal Throttling Benchmark under continuous load.")
    parser.add_argument("--data", default="data/delays_zurich_transport_45034.parquet", help="Path to parquet file")
    parser.add_argument("--output", default="throttling_results.csv", help="CSV output path")
    parser.add_argument("--plot", default="figures/thermal_throttling_runs.png", help="Figure output path")
    parser.add_argument("--runs", type=int, default=15, help="Number of consecutive iterations (default: 15)")
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf", help="Model to train (rf: RandomForest, xgb: XGBoost)")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size (None for full 5.46M dataset)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of CPU cores to utilize (-1 for all)")
    parser.add_argument("--label", default=None, help="Machine label for reporting (e.g. MacBook_Air_M4, Cluster)")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=15, help="Tree max depth")
    parser.add_argument("--show", action="store_true", help="Display plot window upon completion")
    args = parser.parse_args()

    run_throttling_benchmark(
        parquet_path=args.data,
        csv_file=args.output,
        plot_path=args.plot,
        total_runs=args.runs,
        model_type=args.model,
        sample_size=args.sample_size,
        n_jobs=args.n_jobs,
        machine_label=args.label,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        show_plot=args.show,
    )
