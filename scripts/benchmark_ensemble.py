"""
Parallel Ensemble Training via IPyParallel (Day 8 — HPC Cluster & Local)
========================================================================
Implements Phase 3 (Day 8) of the HPC Sprint Roadmap:
Trains diverse sub-models (Random Forest & XGBoost) in parallel across
IPyParallel engines and aggregates them into an ensemble predictor (Voting/Blending).

Demonstrates:
- Distributed model fitting across cluster nodes.
- Reduction in prediction variance (Ensemble RMSE vs. single model RMSE).
- Wall-clock speedup of parallel ensemble training.
"""

import os
import sys
import time
import socket
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def init_worker_data(parquet_path: str, sample_size: Optional[int] = None, random_state: int = 42) -> Dict[str, Any]:
    """
    Worker-side dataset loader. Caches data in process memory.
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
        raise KeyError(f"Target column '{target_col}' not found: {list(df.columns)}")

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
    }


def train_ensemble_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker task: fits a sub-model and returns predictions on the test set.
    """
    import os
    import time
    import builtins
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

    submodel_id = task["submodel_id"]
    model_type = task["model_type"]
    params = dict(task["params"])
    seed = task.get("seed", 42)

    cache = getattr(builtins, "_IPP_WORKER_CACHE", None)
    if cache is None or "X_train" not in cache:
        init_worker_data(parquet_path=task["parquet_path"], sample_size=task["sample_size"], random_state=42)
        cache = builtins._IPP_WORKER_CACHE

    X_train = cache["X_train"]
    y_train = cache["y_train"]
    X_test = cache["X_test"]
    y_test = cache["y_test"]

    t0 = time.time()
    if model_type == "RandomForest":
        model = RandomForestRegressor(**params, n_jobs=1, random_state=seed)
    elif model_type == "XGBoost":
        tree_method = params.pop("tree_method", "hist")
        model = XGBRegressor(**params, tree_method=tree_method, n_jobs=1, random_state=seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    train_time = time.time() - t0

    preds = model.predict(X_test)

    rmse = float(root_mean_squared_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    return {
        "submodel_id": submodel_id,
        "name": f"{model_type}_{submodel_id}",
        "model_type": model_type,
        "train_time_sec": round(train_time, 4),
        "predictions": preds.tolist(),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "worker_pid": os.getpid(),
    }


def run_parallel_ensemble(
    parquet_path: str = "data/delays_zurich_transport_45034.parquet",
    output_csv: str = "results/benchmark_ensemble_results.csv",
    sample_size: Optional[int] = None,
    n_engines: int = 4,
    auto_start_cluster: bool = True,
):
    import ipyparallel as ipp
    from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

    print("=" * 80)
    print(" Parallel Ensemble Regressor (Day 8 — Voting / Blending via IPyParallel)")
    print("=" * 80)

    cluster = None
    try:
        rc = ipp.Client(timeout=4)
        print(f"Connected to existing cluster! Active engines ({len(rc.ids)}): {rc.ids}")
    except Exception as e:
        if not auto_start_cluster:
            raise RuntimeError(f"Could not connect to IPyParallel controller: {e}")
        print(f"Starting local IPyParallel cluster with {n_engines} engines...")
        cluster = ipp.Cluster(n=n_engines)
        rc = cluster.start_and_connect_sync()
        print(f"Local cluster ready with {len(rc.ids)} engines.")

    try:
        engine_count = len(rc.ids)
        dview = rc[:]
        lview = rc.load_balanced_view()

        # Step 1: Pre-cache data on workers
        print("\n[1/3] Broadcasting dataset caching to all engines...")
        t_init0 = time.time()
        dview.apply_sync(init_worker_data, parquet_path, sample_size, 42)
        print(f"Dataset cached on all {engine_count} engines in {time.time() - t_init0:.2f}s.")

        # Step 2: Define diverse ensemble sub-models
        ensemble_configs = [
            {"model_type": "RandomForest", "params": {"n_estimators": 60, "max_depth": 12}, "seed": 10},
            {"model_type": "RandomForest", "params": {"n_estimators": 80, "max_depth": 14}, "seed": 20},
            {"model_type": "RandomForest", "params": {"n_estimators": 100, "max_depth": 16}, "seed": 30},
            {"model_type": "RandomForest", "params": {"n_estimators": 100, "max_depth": 18}, "seed": 40},
            {"model_type": "XGBoost", "params": {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.08, "tree_method": "hist"}, "seed": 11},
            {"model_type": "XGBoost", "params": {"n_estimators": 150, "max_depth": 8, "learning_rate": 0.05, "tree_method": "hist"}, "seed": 22},
            {"model_type": "XGBoost", "params": {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.04, "tree_method": "hist"}, "seed": 33},
            {"model_type": "XGBoost", "params": {"n_estimators": 250, "max_depth": 10, "learning_rate": 0.03, "tree_method": "hist"}, "seed": 44},
        ]

        tasks = []
        for idx, cfg in enumerate(ensemble_configs, start=1):
            tasks.append({
                "submodel_id": idx,
                "model_type": cfg["model_type"],
                "params": cfg["params"],
                "seed": cfg["seed"],
                "parquet_path": parquet_path,
                "sample_size": sample_size,
            })

        print(f"\n[2/3] Training {len(tasks)} ensemble sub-models in parallel across {engine_count} engines...")
        t_start = time.time()
        submodel_results = lview.map_sync(train_ensemble_worker, tasks)
        parallel_wall_clock = time.time() - t_start

        # Step 3: Ensemble Aggregation (Voting / Averaging)
        print("\n[3/3] Aggregating sub-model predictions (Uniform Weighted Ensemble)...")
        all_preds = np.array([r["predictions"] for r in submodel_results])
        ensemble_pred = np.mean(all_preds, axis=0)

        # Retrieve y_test for final metric evaluation
        df_all = pd.read_parquet(parquet_path)
        if sample_size and 0 < sample_size < len(df_all):
            df_all = df_all.sample(n=sample_size, random_state=42)
        from sklearn.model_selection import train_test_split
        _, _, _, y_test = train_test_split(df_all.drop(columns=["delay"]), df_all["delay"], test_size=0.2, random_state=42)
        y_test_arr = y_test.to_numpy()

        ensemble_rmse = float(root_mean_squared_error(y_test_arr, ensemble_pred))
        ensemble_mae = float(mean_absolute_error(y_test_arr, ensemble_pred))
        ensemble_r2 = float(r2_score(y_test_arr, ensemble_pred))

        print("\n" + "=" * 80)
        print(" Individual Sub-Model Performance:")
        print("=" * 80)
        print(f"{'ID':<4} | {'Model Name':<18} | {'Train(s)':<9} | {'RMSE':<8} | {'R^2':<8} | {'Worker PID'}")
        print("-" * 80)
        for r in submodel_results:
            print(
                f"{r['submodel_id']:<4} | "
                f"{r['name']:<18} | "
                f"{r['train_time_sec']:>7.2f}s | "
                f"{r['rmse']:>8.4f} | "
                f"{r['r2']:>8.4f} | "
                f"PID {r['worker_pid']}"
            )

        sum_seq_time = sum(r["train_time_sec"] for r in submodel_results)
        speedup = sum_seq_time / parallel_wall_clock if parallel_wall_clock > 0 else 1.0

        print("=" * 80)
        print(" FINAL ENSEMBLE EVALUATION (VOTING REGRESSOR):")
        print("=" * 80)
        print(f" Ensemble RMSE            : {ensemble_rmse:.4f}")
        print(f" Ensemble MAE             : {ensemble_mae:.4f}")
        print(f" Ensemble R^2             : {ensemble_r2:.4f}")
        print(f" Wall-Clock Parallel Time : {parallel_wall_clock:.2f} seconds")
        print(f" Cumulative Serial Time   : {sum_seq_time:.2f} seconds")
        print(f" Parallel Concurrency     : {speedup:.2f}x speedup across {engine_count} engines")
        print("=" * 80)

        # Save to CSV
        records = []
        for r in submodel_results:
            records.append({
                "type": "submodel",
                "name": r["name"],
                "train_time_sec": r["train_time_sec"],
                "rmse": r["rmse"],
                "r2": r["r2"],
            })
        records.append({
            "type": "ensemble",
            "name": "VotingRegressor_Ensemble",
            "train_time_sec": round(parallel_wall_clock, 4),
            "rmse": round(ensemble_rmse, 4),
            "r2": round(ensemble_r2, 4),
        })
        df_out = pd.DataFrame(records)
        df_out.to_csv(output_csv, index=False)
        print(f"\nEnsemble results successfully exported to: {output_csv}\n")

    finally:
        if cluster is not None:
            print("Stopping local cluster...")
            cluster.stop_cluster_sync()
            print("Local cluster stopped cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Ensemble Training via IPyParallel")
    parser.add_argument("--data", default="data/delays_zurich_transport_45034.parquet", help="Path to parquet file")
    parser.add_argument("--output", default="results/benchmark_ensemble_results.csv", help="CSV output path")
    parser.add_argument("--sample-size", type=int, default=0, help="Row sample size (0 for full dataset)")
    parser.add_argument("--engines", type=int, default=4, help="Local engines to spawn if auto-starting")
    parser.add_argument("--no-auto-start", action="store_true", help="Fail if no controller is running")

    args = parser.parse_args()
    sample_sz = args.sample_size if args.sample_size > 0 else None

    run_parallel_ensemble(
        parquet_path=args.data,
        output_csv=args.output,
        sample_size=sample_sz,
        n_engines=args.engines,
        auto_start_cluster=not args.no_auto_start,
    )
