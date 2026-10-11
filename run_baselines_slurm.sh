#!/usr/bin/bash
# ==============================================================================
# Slurm Batch-Skript für Shared-Memory Multi-Core Baselines (HPC SoSe 2026)
# Exakt abgestimmt auf die Cluster-Konfiguration
# ==============================================================================
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION>

#SBATCH --job-name=baselines_benchmark
#SBATCH --output=%u.log.%j.out
#SBATCH --error=%u.log.%j.err

#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

set -e

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Maximal threads per process: $SLURM_CPUS_PER_TASK"
echo "Current working directory is `pwd`"

# 1. Verzeichnisse anlegen
mkdir -p logs figures data results

# 2. Module laden & Environment aktivieren
echo "[1/4] Lade Module und aktiviere Python-Environment..."
module purge
module add slurm
module add miniconda3 || module load miniconda3 || module load python/3.11 || true

# Umgebung aktivieren (unterstützt Conda 'data-science' oder eigenes venv)
if [ -d "$HOME/.conda/envs/data-science" ]; then
    source activate data-science
elif [ -d "$HOME/.conda/envs/hpc_env" ]; then
    source activate hpc_env
elif [ -f "$HOME/hpc_env/bin/activate" ]; then
    source "$HOME/hpc_env/bin/activate"
elif [ -f "./.venv/bin/activate" ]; then
    source "./.venv/bin/activate"
fi

python3 --version

# 3. Datensatz prüfen
DATASET_PATH="data/delays_zurich_transport_45034.parquet"
if [ ! -f "$DATASET_PATH" ]; then
    echo "Datensatz nicht gefunden. Starte Download..."
    python3 scripts/download_45034.py
fi

# 4. XGBoost Multi-Core Scaling Benchmark (1, 2, 4, 8, 12, 16, 24 Kerne)
echo "[2/4] Starte XGBoost Multi-Core Benchmark (1..24 Kerne)..."
python3 scripts/benchmark_xgboost.py \
    --data "$DATASET_PATH" \
    --output "results/benchmark_cluster_xgboost.csv" \
    --cores 1 2 4 8 12 16 24 \
    --runs 4

python3 scripts/visualize_benchmark.py \
    --csv "results/benchmark_cluster_xgboost.csv" \
    --output "figures/benchmark_cluster_xgboost_median.png" \
    --no-show

# 5. RandomForest Multi-Core Scaling Benchmark (1, 2, 4, 8, 12, 16, 24 Kerne)
echo "[3/4] Starte RandomForest Multi-Core Benchmark (1..24 Kerne)..."
python3 scripts/test.py \
    --data "$DATASET_PATH" \
    --output "results/benchmark_cluster_rf.csv" \
    --cores 1 2 4 8 12 16 24 \
    --runs 4

python3 scripts/visualize_benchmark.py \
    --csv "results/benchmark_cluster_rf.csv" \
    --output "figures/benchmark_cluster_rf_median.png" \
    --no-show

# 6. Thermal Throttling Gegenbeweis auf dem Cluster
echo "[4/4] Starte Cluster Throttling-Messung (Beweis für thermische Stabilität)..."
python3 scripts/benchmark_throttling.py \
    --data "$DATASET_PATH" \
    --runs 15 \
    --n-jobs 24 \
    --output "figures/cluster_thermal_throttling.png" || true

echo "======================================================================"
echo "Finished at `date`"
echo "Ergebnisse gespeichert in: results/benchmark_cluster_*.csv"
echo "Grafiken gespeichert in  : figures/"
echo "======================================================================"
