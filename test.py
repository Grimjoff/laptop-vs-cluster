import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Load dataset
print("Loading dataset...")
df = pd.read_parquet("data/delays_zurich_transport_45034.parquet")
print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

# Set SAMPLE_SIZE to an integer (e.g. 100_000) for fast prototyping, or None for the full 5.46M rows
SAMPLE_SIZE = None

if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
    print(f"Using a sample of {SAMPLE_SIZE:,} rows (set SAMPLE_SIZE = None for all {len(df):,} rows)...")
    data = df.sample(n=SAMPLE_SIZE, random_state=42)
else:
    data = df

# 2. Prepare features and target
target_col = "delay"
X = data.drop(columns=[target_col])
y = data[target_col]

# 3. Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

# Benchmark configuration & hyperparameters
n_estimators = 100
max_depth = 15
random_state = 42
cores_amount = [1, 2, 4, 8, 10, 12, 16]
csv_file = "benchmark_results.csv"

# Ensure output directory exists if a subfolder is specified
csv_dir = os.path.dirname(csv_file)
if csv_dir:
    os.makedirs(csv_dir, exist_ok=True)

print("\nTraining RandomForestRegressor...")
for number in cores_amount:
    for i in range(4):
    # 4. Initialize and train RandomForest Regressor
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=number,          # Utilize specified CPU cores
            random_state=random_state,
        )

        t0 = time.time()
        rf.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"Training on {number} cores completed in {train_time:.2f} seconds.")

        # Save each run's setup and duration to CSV
        run_result = pd.DataFrame([{
            "sample_size": len(data),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "n_jobs": number,
            "duration_sec": round(train_time, 4),
        }])

        file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
        run_result.to_csv(csv_file, mode="a", header=not file_exists, index=False)

y_pred = rf.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# 6. Feature Importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Feature Importances ---")
for feat, imp in importances.items():
    print(f"  {feat:15s}: {imp:.4f}")