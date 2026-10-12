"""
Intel Extension for Scikit-Learn Benchmark (HPC SoSe 2026)
===========================================================
Compares Stock Scikit-Learn against Intel Extension for Scikit-Learn
(scikit-learn-intelex / Intel oneDAL) on multi-core Intel Xeon CPUs.

Specifically designed for the HPC cluster (Dual-Socket Intel Xeon Gold 6226,
24 physical cores, AVX-512 vector units, 192 GB RAM).

Measures:
- Training Duration (Stock Scikit-Learn vs. Intel Extension)
- Inference Duration
- Prediction Accuracy (RMSE, MAE, R²) to prove numerical equivalence
- Hardware-Accelerated Speedup: S = T_stock / T_intelex
- System Architecture Telemetry (CPU model, AVX capabilities, socket info)

Exports:
- CSV: results/benchmark_intelex.csv
- Plot: figures/benchmark_stock_vs_intelex.png
"""

import os
import sys
import time
import socket
import platform
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def get_cpu_telemetry() -> Dict[str, str]:
    """Gathers CPU architecture and platform metadata."""
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    
    # Try reading /proc/cpuinfo on Linux for detailed Xeon telemetry
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
            for line in content.splitlines():
                if "model name" in line:
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
            info["has_avx512"] = str("avx512" in content.lower())
            info["has_avx2"] = str("avx2" in content.lower())
        except Exception:
            pass
    return info


def run_benchmark(
    parquet_path: str = "data/delays_zurich_transport_45034.parquet",
    output_csv: str = "results/benchmark_intelex.csv",
    output_plot: str = "figures/benchmark_stock_vs_intelex.png",
    cores_list: Optional[List[int]] = None,
    runs_per_config: int = 3,
    n_estimators: int = 100,
    max_depth: int = 15,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    generate_plot: bool = True,
):
    if cores_list is None:
        cores_list = [24]

    telemetry = get_cpu_telemetry()
    cpu_label = telemetry.get("cpu_model", telemetry.get("processor", telemetry["machine"]))

    print("=" * 80)
    print(" Intel Extension for Scikit-Learn (scikit-learn-intelex) Benchmark")
    print("=" * 80)
    print(f"Hostname     : {telemetry['hostname']}")
    print(f"Architecture : {telemetry['machine']} ({cpu_label})")
    if "has_avx512" in telemetry:
        print(f"AVX-512 Flags: {telemetry['has_avx512']} | AVX2 Flags: {telemetry['has_avx2']}")
    print(f"Dataset      : {parquet_path}")
    print(f"Tested Cores : {cores_list} | Runs per config: {runs_per_config}")
    print(f"Hyperparams  : n_estimators={n_estimators}, max_depth={max_depth}")
    print("=" * 80)

    # 1. Check for sklearnex availability
    intelex_available = False
    try:
        import sklearnex
        intelex_available = True
        print(f"[+] Intel Extension for Scikit-Learn detected (v{sklearnex.__version__})")
    except ImportError:
        print("[-] 'scikit-learn-intelex' is NOT installed in this environment.")
        if telemetry["machine"] == "arm64":
            print("    -> Note: Apple Silicon (ARM64) does not support Intel oneDAL AVX-512.")
            print("    -> Execute this benchmark on the HPC cluster (Intel Xeon Gold)!")
        else:
            print("    -> Install on cluster with: pip install scikit-learn-intelex")

    # 2. Load dataset
    print(f"\n[1/3] Loading dataset...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Dataset file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    total_len = len(df)
    print(f"  -> Total dataset rows: {total_len:,}")

    if sample_size and 0 < sample_size < total_len:
        print(f"  -> Downsampling to {sample_size:,} rows for rapid profiling...")
        df = df.sample(n=sample_size, random_state=random_state)

    target_col = "delay"
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    print(f"  -> Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")

    records = []

    # 3. Benchmark Loop across core configurations
    for cores in cores_list:
        print(f"\n[2/3] Benchmarking with {cores} CPU Cores (n_jobs={cores})...")

        # ------------------------------------------------------------------
        # Variant A: Stock Scikit-Learn
        # ------------------------------------------------------------------
        print(f"\n--- [A] Stock Scikit-Learn (Pure C / OpenMP) ---")
        from sklearn.ensemble import RandomForestRegressor as StockRF

        stock_durations = []
        stock_rmses = []
        for run_idx in range(1, runs_per_config + 1):
            rf_stock = StockRF(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=cores,
                random_state=random_state + run_idx,
            )
            t0 = time.time()
            rf_stock.fit(X_train, y_train)
            train_time = time.time() - t0

            t_pred0 = time.time()
            y_pred = rf_stock.predict(X_test)
            pred_time = time.time() - t_pred0
            total_duration = train_time + pred_time

            rmse = float(root_mean_squared_error(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            stock_durations.append(total_duration)
            stock_rmses.append(rmse)
            print(f"  Run {run_idx}/{runs_per_config}: Train={train_time:6.2f}s | Pred={pred_time:5.2f}s | RMSE={rmse:.4f}")

            records.append({
                "implementation": "Stock Scikit-Learn",
                "cores": cores,
                "run": run_idx,
                "dataset_rows": len(df),
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "train_time_sec": round(train_time, 4),
                "pred_time_sec": round(pred_time, 4),
                "duration_sec": round(total_duration, 4),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2": round(r2, 4),
                "hostname": telemetry["hostname"],
                "cpu_model": cpu_label,
            })

        avg_stock_duration = np.mean(stock_durations)
        print(f"  ==> Stock Scikit-Learn Average Duration: {avg_stock_duration:.2f}s")

        # ------------------------------------------------------------------
        # Variant B: Intel Extension for Scikit-Learn (oneDAL / AVX-512)
        # ------------------------------------------------------------------
        if intelex_available:
            print(f"\n--- [B] Intel Extension for Scikit-Learn (Intel oneDAL AVX-512) ---")
            try:
                from sklearnex.ensemble import RandomForestRegressor as IntelRF
            except (ImportError, AttributeError):
                from sklearnex import patch_sklearn
                patch_sklearn()
                from sklearn.ensemble import RandomForestRegressor as IntelRF

            intel_durations = []
            intel_rmses = []
            for run_idx in range(1, runs_per_config + 1):
                rf_intel = IntelRF(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    n_jobs=cores,
                    random_state=random_state + run_idx,
                )
                t0 = time.time()
                rf_intel.fit(X_train, y_train)
                train_time = time.time() - t0

                t_pred0 = time.time()
                y_pred = rf_intel.predict(X_test)
                pred_time = time.time() - t_pred0
                total_duration = train_time + pred_time

                rmse = float(root_mean_squared_error(y_test, y_pred))
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))

                intel_durations.append(total_duration)
                intel_rmses.append(rmse)
                print(f"  Run {run_idx}/{runs_per_config}: Train={train_time:6.2f}s | Pred={pred_time:5.2f}s | RMSE={rmse:.4f}")

                records.append({
                    "implementation": "Intel Extension (oneDAL)",
                    "cores": cores,
                    "run": run_idx,
                    "dataset_rows": len(df),
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "train_time_sec": round(train_time, 4),
                    "pred_time_sec": round(pred_time, 4),
                    "duration_sec": round(total_duration, 4),
                    "rmse": round(rmse, 4),
                    "mae": round(mae, 4),
                    "r2": round(r2, 4),
                    "hostname": telemetry["hostname"],
                    "cpu_model": cpu_label,
                })

            avg_intel_duration = np.mean(intel_durations)
            speedup = avg_stock_duration / avg_intel_duration
            print(f"  ==> Intel Extension Average Duration: {avg_intel_duration:.2f}s")
            print(f"  >>> OBSERVED SPEEDUP VIA INTEL ONEDAL: {speedup:.2f}x <<<")
        else:
            print("\n--- [B] Intel Extension skipped (sklearnex not installed or on ARM64) ---")

    # 4. Save CSV
    df_results = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    df_results.to_csv(output_csv, mode="a", header=not file_exists, index=False)
    print(f"\n[3/3] Results saved to: {output_csv}")

    # 5. Generate Publication Plot
    if generate_plot and intelex_available and len(records) > 0:
        plot_intelex_comparison(df_results, output_plot)


def plot_intelex_comparison(df: pd.DataFrame, output_path: str):
    """Generates dual-panel publication-grade figure comparing Stock Sklearn vs. Intel Extension."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    summary = df.groupby(["implementation", "cores"]).agg({
        "duration_sec": "mean",
        "train_time_sec": "mean",
        "pred_time_sec": "mean",
        "rmse": "mean",
    }).reset_index()

    stock_row = summary[summary["implementation"] == "Stock Scikit-Learn"].iloc[0]
    intel_row = summary[summary["implementation"] == "Intel Extension (oneDAL)"].iloc[0]

    t_stock = stock_row["duration_sec"]
    t_intel = intel_row["duration_sec"]
    speedup = t_stock / t_intel

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=300)

    # Panel 1: Runtimes
    categories = ["Stock Scikit-Learn\n(Pure C / OpenMP)", "Intel Extension\n(Intel oneDAL / AVX-512)"]
    times = [t_stock, t_intel]
    colors = ["#475569", "#0284c7"]

    bars = ax1.bar(categories, times, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.2, zorder=3)
    ax1.set_ylabel("Gesamtlaufzeit in Sekunden (Train + Pred)", fontsize=11, labelpad=10)
    ax1.set_title("A: Trainingsdauer auf Intel Xeon Gold 6226 (24 Kerne)", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.6, zorder=0)

    # Annotate bars
    for bar, t in zip(bars, times):
        ax1.annotate(
            f"{t:.1f}s",
            xy=(bar.get_x() + bar.get_width() / 2, t),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#0f172a",
        )

    # Annotate Speedup badge
    ax1.text(
        0.5, max(times) * 0.85,
        f"Speedup: {speedup:.2f}x schneller\n(Intel oneDAL AVX-512)",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#dcfce7", edgecolor="#16a34a", linewidth=1.5),
        fontsize=10,
        fontweight="bold",
        color="#166534",
    )

    # Panel 2: Accuracy Verification (Numerical Equivalence)
    rmses = [stock_row["rmse"], intel_row["rmse"]]
    bars2 = ax2.bar(categories, rmses, color=["#64748b", "#38bdf8"], width=0.55, edgecolor="#0f172a", linewidth=1.2, zorder=3)
    ax2.set_ylabel("Test-Set Root Mean Squared Error (RMSE)", fontsize=11, labelpad=10)
    ax2.set_title("B: Modellgüte-Verifikation (Numerische Äquivalenz)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylim(0, max(rmses) * 1.25)
    ax2.grid(True, linestyle="--", alpha=0.6, zorder=0)

    for bar, r in zip(bars2, rmses):
        ax2.annotate(
            f"RMSE: {r:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, r),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
        )

    fig.suptitle(
        f"Vergleich: Standard Scikit-Learn vs. Intel Extension for Scikit-Learn (Ref. [2])\nDatensatz: delays_zurich_transport ({stock_row['cores']:.0f} Cores, 192 GB RAM)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafik erfolgreich generiert: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Stock Sklearn vs. Intel Extension (oneDAL)")
    parser.add_argument("--data", default="data/delays_zurich_transport_45034.parquet", help="Path to parquet file")
    parser.add_argument("--output", default="results/benchmark_intelex.csv", help="CSV output path")
    parser.add_argument("--plot-path", default="figures/benchmark_stock_vs_intelex.png", help="Plot output path")
    parser.add_argument("--cores", nargs="+", type=int, default=[24], help="List of core counts to test (e.g. 24)")
    parser.add_argument("--runs", type=int, default=3, help="Repetitions per configuration")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=15, help="Max tree depth")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional row count (None/0 for all 5.46M rows)")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate plot")

    args = parser.parse_args()

    sample_sz = args.sample_size if args.sample_size and args.sample_size > 0 else None

    run_benchmark(
        parquet_path=args.data,
        output_csv=args.output,
        output_plot=args.plot_path,
        cores_list=args.cores,
        runs_per_config=args.runs,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        sample_size=sample_sz,
        generate_plot=not args.no_plot,
    )
