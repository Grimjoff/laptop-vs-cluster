import os
import time
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor


def run_xgboost_benchmark(
    parquet_path="data/delays_zurich_transport_45034.parquet",
    csv_file="results/benchmark_results_xgboost.csv",
    cores_list=None,
    runs_per_core=4,
    n_estimators=100,
    max_depth=8,
    sample_size=None,
    random_state=42,
):
    if cores_list is None:
        cores_list = [1, 2, 4, 8, 10, 12, 16]

    # 1. Load dataset
    print(f"Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)
    print(f"Loaded {total_rows:,} rows and {len(df.columns)} columns.")

    if sample_size and sample_size < total_rows:
        print(f"Using a sample of {sample_size:,} rows...")
        data = df.sample(n=sample_size, random_state=random_state)
    else:
        data = df

    # 2. Prepare features and target
    target_col = "delay"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # 3. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    print(f"Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples\n")

    # Ensure output directory exists
    csv_dir = os.path.dirname(csv_file)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    print("=" * 65)
    print(f"Starting XGBoost Benchmark (tree_method='hist')")
    print(f"Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}")
    print(f"Tested cores: {cores_list} | Runs per setup: {runs_per_core}")
    print("=" * 65)

    last_model = None

    for cores in cores_list:
        print(f"\n>>> Benchmarking n_jobs = {cores} ({runs_per_core} runs) <<<")
        for run_idx in range(1, runs_per_core + 1):
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                tree_method="hist",  # High-performance histogram split method
                n_jobs=cores,        # OpenMP parallel thread count
                random_state=random_state,
            )

            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            last_model = model

            print(f"  Run {run_idx}/{runs_per_core} on {cores:2d} cores: {train_time:6.2f}s")

            # Save each run to CSV
            run_result = pd.DataFrame([{
                "model": "XGBoost",
                "sample_size": len(data),
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state,
                "n_jobs": cores,
                "run": run_idx,
                "duration_sec": round(train_time, 4),
            }])

            file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
            run_result.to_csv(csv_file, mode="a", header=not file_exists, index=False)

    # 4. Evaluation on test set
    if last_model is not None:
        print("\n" + "=" * 65)
        print("Model Evaluation on Test Set (Last Trained Model)")
        print("=" * 65)
        y_pred = last_model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE : {rmse:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"R²   : {r2:.4f}")

        # Feature importances
        importances = pd.Series(last_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\nFeature Importances:")
        for feat, imp in importances.items():
            print(f"  {feat:15s}: {imp:.4f}")

    print(f"\nAll benchmark results successfully appended to: {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark XGBoost parallel scaling across CPU cores.")
    parser.add_argument("--data", default="data/delays_zurich_transport_45034.parquet", help="Path to parquet file")
    parser.add_argument("--output", default="results/benchmark_results_xgboost.csv", help="CSV output path")
    parser.add_argument("--cores", nargs="+", type=int, default=[1, 2, 4, 8, 10, 12, 16], help="List of core counts to test")
    parser.add_argument("--runs", type=int, default=4, help="Number of repetitions per core count")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of gradient boosted trees")
    parser.add_argument("--max-depth", type=int, default=8, help="Max tree depth")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional downsampling (None for all 5.46M rows)")
    args = parser.parse_args()

    run_xgboost_benchmark(
        parquet_path=args.data,
        csv_file=args.output,
        cores_list=args.cores,
        runs_per_core=args.runs,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        sample_size=args.sample_size,
    )
