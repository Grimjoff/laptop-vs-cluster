"""
Parallel Model Training with IPyParallel (Random Forest & XGBoost)
-----------------------------------------------------------------
Designed for local multi-engine prototyping and high-performance
benchmarking on HPC clusters (e.g. Slurm with 24+ CPU cores).

Features:
- Connects automatically to an existing cluster (e.g. Slurm-managed ipcontroller/ipengine)
  or starts a local IPyParallel cluster if none is running.
- In-memory data caching on workers to eliminate ZeroMQ serialization bottlenecks.
- Asynchronous task distribution across engines via LoadBalancedView.
- Trains RandomForestRegressor and XGBRegressor in parallel.
- Computes regression metrics (RMSE, MAE, R²) and execution times.
- Exports results to a standardized CSV matching team benchmark specifications.
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
    Initializes and caches the dataset in the worker's process memory.
    This runs once per IPyParallel engine so large datasets are NOT transferred
    repeatedly over network sockets during task distribution.
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
        raise KeyError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

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
    Task function executed on an IPyParallel engine.
    Trains either RandomForestRegressor or XGBRegressor and evaluates predictions.
    """
    import os
    import time
    import socket
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

    # Lazy initialization fallback in case worker wasn't pre-initialized
    cache = getattr(builtins, "_IPP_WORKER_CACHE", None)
    if cache is None or "X_train" not in cache:
        init_worker_data(parquet_path=parquet_path, sample_size=sample_size, random_state=random_state)
        cache = builtins._IPP_WORKER_CACHE

    X_train = cache["X_train"]
    y_train = cache["y_train"]
    X_test = cache["X_test"]
    y_test = cache["y_test"]

    # Model instantiation
    t_start = time.time()
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

    # Training
    t0 = time.time()
    model.fit(X_train, y_train)
    train_duration = time.time() - t0

    # Prediction / Evaluation
    t_pred0 = time.time()
    y_pred = model.predict(X_test)
    pred_duration = time.time() - t_pred0
    total_duration = time.time() - t_start

    rmse = float(root_mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    return {
        "task_id": task_id,
        "model_type": model_type,
        "params": str(params),
        "n_estimators": params.get("n_estimators", None),
        "max_depth": params.get("max_depth", None),
        "learning_rate": params.get("learning_rate", None),
        "n_jobs_per_worker": n_jobs,
        "sample_size": cache.get("sample_size", len(X_train) + len(X_test)),
        "train_time_sec": round(train_duration, 4),
        "pred_time_sec": round(pred_duration, 4),
        "total_duration_sec": round(total_duration, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "hostname": socket.gethostname(),
        "worker_pid": os.getpid(),
    }


def connect_or_start_cluster(n_engines: int = 4, timeout: int = 5, auto_start: bool = True) -> Tuple[Any, Optional[Any]]:
    """
    Connects to an existing IPyParallel cluster if available.
    If no controller is found and auto_start is True, starts a local cluster.
    """
    import ipyparallel as ipp

    print("[1/4] Connecting to IPyParallel Cluster...")
    try:
        rc = ipp.Client(timeout=timeout)
        print(f"  -> Connected to existing cluster! Active engines ({len(rc.ids)}): {rc.ids}")
        return rc, None
    except Exception as e:
        if not auto_start:
            print(f"  -> No existing cluster detected: {e}")
            raise RuntimeError(
                "Could not connect to an existing IPyParallel controller. "
                "Start one with 'ipcluster start -n <cores>' or run within a Slurm job."
            )

        print(f"  -> No running cluster found ({e.__class__.__name__}).")
        print(f"  -> Auto-starting a local IPyParallel cluster with {n_engines} engines...")
        cluster = ipp.Cluster(n=n_engines)
        rc = cluster.start_and_connect_sync()
        print(f"  -> Local cluster ready! Active engines ({len(rc.ids)}): {rc.ids}")
        return rc, cluster


def generate_benchmark_tasks(models: str = "all", runs_per_config: int = 1) -> List[Dict[str, Any]]:
    """
    Builds a list of training configurations for Random Forest and XGBoost.
    """
    configs: List[Tuple[str, Dict[str, Any]]] = []

    rf_grid = [
        {"n_estimators": 50, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 15},
        {"n_estimators": 100, "max_depth": 20},
    ]

    xgb_grid = [
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "tree_method": "hist"},
        {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.05, "tree_method": "hist"},
        {"n_estimators": 300, "max_depth": 10, "learning_rate": 0.05, "tree_method": "hist"},
    ]

    if models in ("all", "rf"):
        for cfg in rf_grid:
            configs.append(("RandomForest", cfg))

    if models in ("all", "xgb"):
        for cfg in xgb_grid:
            configs.append(("XGBoost", cfg))

    tasks = []
    task_counter = 1
    for run_idx in range(1, runs_per_config + 1):
        for model_type, params in configs:
            tasks.append({
                "task_id": task_counter,
                "model_type": model_type,
                "params": params,
                "run": run_idx,
            })
            task_counter += 1

    return tasks


def run_ipp_parallel_training(
    parquet_path: str = "data/delays_zurich_transport_45034.parquet",
    output_csv: str = "results/benchmark_results_ipp.csv",
    sample_size: Optional[int] = 100_000,
    models: str = "all",
    n_engines: int = 4,
    n_jobs_per_worker: int = 1,
    runs: int = 1,
    auto_start_cluster: bool = True,
):
    """
    Main orchestration routine for parallel model training across IPyParallel engines.
    """
    rc, cluster = connect_or_start_cluster(
        n_engines=n_engines,
        timeout=4,
        auto_start=auto_start_cluster,
    )

    try:
        engine_count = len(rc.ids)
        if engine_count == 0:
            raise RuntimeError("Connected to IPyParallel client, but 0 engines are registered!")

        dview = rc[:]
        lview = rc.load_balanced_view()

        # Step 1: Pre-initialize data on each worker
        print(f"\n[2/4] Broadcasting data initialization to {engine_count} engines...")
        print(f"  Parquet path: {parquet_path}")
        print(f"  Sample size : {sample_size if sample_size else 'Full Dataset (~5.46M rows)'}")
        t_init0 = time.time()
        init_reports = dview.apply_sync(init_worker_data, parquet_path, sample_size, 42)
        init_time = time.time() - t_init0
        print(f"  -> All engines loaded data in {init_time:.2f}s:")
        for rep in init_reports:
            print(f"     PID {rep['pid']:5d} on {rep['hostname']} loaded {rep['samples_loaded']:,} samples ({rep['features']} features)")

        # Step 2: Build parallel task queue
        tasks = generate_benchmark_tasks(models=models, runs_per_config=runs)
        for t in tasks:
            t["n_jobs"] = n_jobs_per_worker
            t["parquet_path"] = parquet_path
            t["sample_size"] = sample_size
            t["random_state"] = 42

        print(f"\n[3/4] Distributing {len(tasks)} model training tasks via LoadBalancedView across {engine_count} engines...")
        print("=" * 82)
        print(f"{'Task':<6} | {'Model':<13} | {'Hyperparameters':<32} | {'Worker PID':<10} | {'Status'}")
        print("=" * 82)
        for t in tasks:
            print(f"{t['task_id']:<6} | {t['model_type']:<13} | {str(t['params']):<32} | {'pending':<10} | In queue")
        print("=" * 82)

        # Step 3: Run parallel training
        t_all0 = time.time()
        print("\n>>> Parallel execution running... Please wait <<<")
        results = lview.map_sync(train_worker_task, tasks)
        wall_clock_time = time.time() - t_all0

        # Step 4: Summary Table and CSV Export
        print("\n[4/4] Parallel Benchmark Results:")
        print("=" * 95)
        print(f"{'ID':<4} | {'Model':<13} | {'Train(s)':<9} | {'Pred(s)':<8} | {'Total(s)':<9} | {'RMSE':<7} | {'R^2':<7} | {'Worker PID'}")
        print("=" * 95)

        records = []
        for res in results:
            print(
                f"{res['task_id']:<4} | "
                f"{res['model_type']:<13} | "
                f"{res['train_time_sec']:>7.2f}s | "
                f"{res['pred_time_sec']:>6.2f}s | "
                f"{res['total_duration_sec']:>7.2f}s | "
                f"{res['rmse']:>7.4f} | "
                f"{res['r2']:>7.4f} | "
                f"PID {res['worker_pid']}"
            )

            # Record adhering to project benchmark conventions
            records.append({
                "method": "ipyparallel",
                "model": res["model_type"],
                "cores": engine_count,
                "dataset_rows": res["sample_size"],
                "n_estimators": res["n_estimators"],
                "max_depth": res["max_depth"],
                "learning_rate": res["learning_rate"],
                "n_jobs_per_worker": res["n_jobs_per_worker"],
                "train_time_sec": res["train_time_sec"],
                "pred_time_sec": res["pred_time_sec"],
                "duration_sec": res["total_duration_sec"],
                "rmse": res["rmse"],
                "mae": res["mae"],
                "r2": res["r2"],
                "worker_pid": res["worker_pid"],
            })

        df_results = pd.DataFrame(records)

        # Ensure directory exists and append/write to CSV
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
        file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
        df_results.to_csv(output_csv, mode="a", header=not file_exists, index=False)

        sum_train_times = sum(r['train_time_sec'] for r in records)
        theoretical_speedup = sum_train_times / wall_clock_time if wall_clock_time > 0 else 1.0

        print("=" * 95)
        print(f"Wall-Clock Parallel Duration: {wall_clock_time:.2f} seconds")
        print(f"Cumulative Sequential Work   : {sum_train_times:.2f} seconds")
        print(f"Observed Throughput Speedup : {theoretical_speedup:.2f}x (across {engine_count} engines)")
        print(f"Results appended to          : {output_csv}\n")

    finally:
        # Step 5: Clean shutdown if we spun up a local cluster instance
        if cluster is not None:
            print("Stopping locally launched IPyParallel cluster...")
            cluster.stop_cluster_sync()
            print("Local cluster stopped cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Random Forest & XGBoost Training via IPyParallel")
    parser.add_argument("--data", default="data/delays_zurich_transport_45034.parquet", help="Path to Zurich delays dataset")
    parser.add_argument("--output", default="results/benchmark_results_ipp.csv", help="Output CSV path for benchmark metrics")
    parser.add_argument("--sample-size", type=int, default=100_000, help="Row sample count (set 0 or omit on cluster for full 5.46M rows)")
    parser.add_argument("--models", choices=["all", "rf", "xgb"], default="all", help="Model family to benchmark")
    parser.add_argument("--engines", type=int, default=4, help="Engines to spawn if auto-starting local cluster")
    parser.add_argument("--n-jobs", type=int, default=1, help="Threads (n_jobs) per worker process (recommended 1)")
    parser.add_argument("--runs", type=int, default=1, help="Repetitions per configuration")
    parser.add_argument("--no-auto-start", action="store_true", help="Do not auto-start local cluster; fail if no controller found")

    args = parser.parse_args()

    sample_sz = args.sample_size if args.sample_size > 0 else None

    run_ipp_parallel_training(
        parquet_path=args.data,
        output_csv=args.output,
        sample_size=sample_sz,
        models=args.models,
        n_engines=args.engines,
        n_jobs_per_worker=args.n_jobs,
        runs=args.runs,
        auto_start_cluster=not args.no_auto_start,
    )
