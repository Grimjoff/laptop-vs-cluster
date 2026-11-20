"""
Strong Scaling Benchmark via IPyParallel (HPC Cluster & Local)
======================================================================
Measures Strong Scaling of distributed task execution across varying core
counts (e.g. 1, 2, 4, 8, 12, 16, 24 cores) on a fixed total workload.

Calculates:
- Execution Time: T_p
- Speedup       : S(p) = T_1 / T_p
- Efficiency    : E(p) = S(p) / p

Exports to the standardized CSV schema for results:
method,model,cores,dataset_rows,duration_sec,speedup,efficiency,rmse,r2
"""

import os
import sys
import time
import socket
import argparse
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


def init_worker_data(parquet_path: str, sample_size: Optional[int] = None, random_state: int = 42) -> Dict[str, Any]:
    """
    Worker-side dataset loader. Caches data in process memory so zero DataFrame
    transfer is required over ZeroMQ during task dispatch.
    """
    import os
    import socket
    import builtins
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Worker {os.getpid()} could not find file: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    total_len = len(df)

    if sample_size and 0 < sample_size < total_len:
        df = df.sample(n=sample_size, random_state=random_state)

    target_col = "delay"
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    if not hasattr(builtins, "_IPP_WORKER_CACHE"):
        builtins._IPP_WORKER_CACHE = {}

    builtins._IPP_WORKER_CACHE["X_train"] = X_train
    builtins._IPP_WORKER_CACHE["y_train"] = y_train
    builtins._IPP_WORKER_CACHE["X_test"] = X_test
    builtins._IPP_WORKER_CACHE["y_test"] = y_test
    builtins._IPP_WORKER_CACHE["sample_size"] = len(df)
    builtins._IPP_WORKER_CACHE["parquet_path"] = parquet_path

    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "samples_loaded": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X.shape[1],
    }


def train_worker_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a single model fit & predict task on an engine.
    """
    import os
    import time
    import builtins
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

    task_id = task.get("task_id", 0)
    model_type = task.get("model_type", "RandomForest")
    params = dict(task.get("params", {}))
    n_jobs = task.get("n_jobs", 1)
    parquet_path = task.get("parquet_path", "data/delays_zurich_transport_45034.parquet")
    sample_size = task.get("sample_size", None)
    random_state = task.get("random_state", 42)

    cache = getattr(builtins, "_IPP_WORKER_CACHE", None)
    if cache is None or "X_train" not in cache:
        init_worker_data(parquet_path=parquet_path, sample_size=sample_size, random_state=random_state)
        cache = builtins._IPP_WORKER_CACHE

    X_train = cache["X_train"]
    y_train = cache["y_train"]
    X_test = cache["X_test"]
    y_test = cache["y_test"]

    t0 = time.time()
    if model_type == "RandomForest":
        model = RandomForestRegressor(
            **params,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif model_type == "XGBoost":
        tree_method = params.pop("tree_method", "hist")
        model = XGBRegressor(
            **params,
            tree_method=tree_method,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    train_duration = time.time() - t0

    t_pred = time.time()
    y_pred = model.predict(X_test)
    pred_duration = time.time() - t_pred

    rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    return {
        "task_id": task_id,
        "model_type": model_type,
        "train_time_sec": round(train_duration, 4),
        "pred_time_sec": round(pred_duration, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "pid": os.getpid(),
    }


def build_fixed_workload(total_tasks: int = 24, model_choice: str = "both") -> List[Dict[str, Any]]:
    """
    Builds a fixed set of tasks (workload W = const) to measure Strong Scaling across cores.
    Default: 24 tasks so at 24 cores, exactly 1 task per core runs concurrently.
    """
    tasks = []
    for i in range(total_tasks):
        if model_choice == "rf":
            mtype = "RandomForest"
            params = {"n_estimators": 50, "max_depth": 10 + (i % 5)}
        elif model_choice == "xgb":
            mtype = "XGBoost"
            params = {"n_estimators": 100, "max_depth": 6 + (i % 4), "learning_rate": 0.05, "tree_method": "hist"}
        else:
            # Alternating RF and XGBoost
            if i % 2 == 0:
                mtype = "RandomForest"
                params = {"n_estimators": 50, "max_depth": 10 + ((i // 2) % 4)}
            else:
                mtype = "XGBoost"
                params = {"n_estimators": 100, "max_depth": 6 + ((i // 2) % 4), "learning_rate": 0.05, "tree_method": "hist"}

        tasks.append({
            "task_id": i + 1,
            "model_type": mtype,
            "params": params,
        })
    return tasks


def run_strong_scaling_benchmark(
    parquet_path: str = "data/delays_zurich_transport_45034.parquet",
    output_csv: str = "results/benchmark_strong_scaling.csv",
    sample_size: Optional[int] = None,
    cores_list: Optional[List[int]] = None,
    total_tasks: int = 24,
    model_choice: str = "both",
    runs_per_core: int = 1,
    auto_start_cluster: bool = True,
):
    import ipyparallel as ipp

    if cores_list is None:
        cores_list = [1, 2, 4, 8, 12, 16, 24]

    print("=" * 78)
    print(" IPyParallel Strong Scaling Benchmark (HPC SoSe 2026)")
    print("=" * 78)

    # 1. Connect to cluster or start locally
    cluster = None
    try:
        rc = ipp.Client(timeout=15)
        print(f"Connected to existing IPyParallel Cluster! Total registered engines: {len(rc.ids)}")
    except Exception as e:
        if not auto_start_cluster:
            raise RuntimeError(f"Could not connect to IPyParallel: {e}")
        max_c = max(cores_list)
        print(f"No running cluster detected. Auto-starting local cluster with {max_c} engines...")
        cluster = ipp.Cluster(n=max_c)
        rc = cluster.start_and_connect_sync()
        print(f"Local cluster online with {len(rc.ids)} engines.")

    try:
        available_engines = len(rc.ids)
        # Filter cores to those that do not exceed available engines
        active_core_levels = [c for c in cores_list if c <= available_engines]
        if not active_core_levels:
            active_core_levels = [available_engines]

        print(f"\nAvailable engines: {available_engines}")
        print(f"Target core levels: {active_core_levels}")
        print(f"Fixed workload    : {total_tasks} total tasks (W = const)")
        print(f"Sample size       : {sample_size if sample_size else 'Full Dataset (~5.46M rows)'}")

        # 2. Broadcast data loading once to ALL engines
        print("\nBroadcasting in-memory dataset cache to all available engines...")
        dview = rc[:]
        t_init0 = time.time()
        dview.apply_sync(init_worker_data, parquet_path, sample_size, 42)
        print(f"Dataset cached on all {available_engines} engines in {time.time() - t_init0:.2f}s.\n")

        # 3. Iterate through core levels
        baseline_t1: Optional[float] = None
        results_records: List[Dict[str, Any]] = []

        print("=" * 86)
        print(f"{'Cores':<7} | {'Run':<5} | {'Duration (s)':<14} | {'Speedup S(p)':<14} | {'Efficiency E(p)':<16} | {'Avg RMSE'}")
        print("=" * 86)

        for cores in active_core_levels:
            # Subset view utilizing exactly `cores` targets
            target_engine_ids = rc.ids[:cores]
            lview = rc.load_balanced_view(targets=target_engine_ids)

            durations = []
            rmses = []
            r2s = []

            for run_idx in range(1, runs_per_core + 1):
                tasks = build_fixed_workload(total_tasks=total_tasks, model_choice=model_choice)
                for t in tasks:
                    t["parquet_path"] = parquet_path
                    t["sample_size"] = sample_size
                    t["n_jobs"] = 1
                    t["random_state"] = 42

                t_start = time.time()
                task_results = lview.map_sync(train_worker_task, tasks)
                t_duration = time.time() - t_start

                durations.append(t_duration)
                avg_rmse = sum(r["rmse"] for r in task_results) / len(task_results)
                avg_r2 = sum(r["r2"] for r in task_results) / len(task_results)
                rmses.append(avg_rmse)
                r2s.append(avg_r2)

            median_duration = sorted(durations)[len(durations) // 2]
            mean_rmse = sum(rmses) / len(rmses)
            mean_r2 = sum(r2s) / len(r2s)

            if baseline_t1 is None or cores == 1:
                baseline_t1 = median_duration

            speedup = baseline_t1 / median_duration if median_duration > 0 else 1.0
            efficiency = speedup / cores if cores > 0 else 1.0

            print(
                f"{cores:<7d} | "
                f"{runs_per_core:<5d} | "
                f"{median_duration:>12.2f}s | "
                f"{speedup:>12.2f}x | "
                f"{efficiency * 100:>14.1f}% | "
                f"{mean_rmse:>8.4f}"
            )

            # Record in the standardized CSV format
            label_model = "RandomForest" if model_choice == "rf" else ("XGBoost" if model_choice == "xgb" else "EnsembleGrid")
            results_records.append({
                "method": "ipyparallel",
                "model": label_model,
                "cores": cores,
                "dataset_rows": sample_size if sample_size else 5465575,
                "duration_sec": round(median_duration, 4),
                "speedup": round(speedup, 4),
                "efficiency": round(efficiency, 4),
                "rmse": round(mean_rmse, 4),
                "r2": round(mean_r2, 4),
            })

        print("=" * 86)

        # 4. Save results to CSV
        df = pd.DataFrame(results_records)
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nStrong Scaling results successfully exported to: {output_csv}")

    finally:
        if cluster is not None:
            print("Stopping local cluster...")
            cluster.stop_cluster_sync()
            print("Local cluster stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Strong Scaling benchmark across varying core counts.")
    parser.add_argument("--data", default="data/delays_zurich_transport_45034.parquet", help="Path to parquet dataset")
    parser.add_argument("--output", default="results/benchmark_strong_scaling.csv", help="Output CSV path")
    parser.add_argument("--sample-size", type=int, default=0, help="Row sample size (0 for full 5.46M dataset)")
    parser.add_argument("--cores", nargs="+", type=int, default=[1, 2, 4, 8, 12, 16, 24], help="List of core counts")
    parser.add_argument("--tasks", type=int, default=24, help="Fixed workload total task count")
    parser.add_argument("--models", choices=["both", "rf", "xgb"], default="both", help="Models to benchmark")
    parser.add_argument("--runs", type=int, default=1, help="Repetitions per core count")
    parser.add_argument("--no-auto-start", action="store_true", help="Fail if no controller is found")

    args = parser.parse_args()
    sample_sz = args.sample_size if args.sample_size > 0 else None

    run_strong_scaling_benchmark(
        parquet_path=args.data,
        output_csv=args.output,
        sample_size=sample_sz,
        cores_list=args.cores,
        total_tasks=args.tasks,
        model_choice=args.models,
        runs_per_core=args.runs,
        auto_start_cluster=not args.no_auto_start,
    )
