#!/usr/bin/bash
# ==============================================================================
# Slurm Batch-Skript für XGBoost Multi-Core Scaling Benchmark (HPC SoSe 2026)
# Misst echte Thread-Skalierung über 1, 2, 4, 8, 12, 16, 24 Kerne
# ==============================================================================
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION>

#SBATCH --job-name=xgboost_benchmark
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --time=0-00:30:00
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
echo "[1/3] Lade Module und aktiviere Python-Environment..."
module purge
module add slurm
module add miniconda3 || module load miniconda3 || module load python/3.11 || true

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
    echo "[2/3] Datensatz nicht gefunden. Starte Download..."
    python3 scripts/download_45034.py
fi

# 4. XGBoost Multi-Core Scaling Benchmark (1, 2, 4, 8, 12, 16, 24 Kerne)
echo "[3/3] Starte XGBoost Multi-Core Benchmark (1..24 Kerne)..."
python3 scripts/benchmark_xgboost.py \
    --data "$DATASET_PATH" \
    --output "results/benchmark_cluster_xgboost.csv" \
    --cores 1 2 4 8 12 16 24 \
    --runs 3

python3 scripts/visualize_benchmark.py \
    --csv "results/benchmark_cluster_xgboost.csv" \
    --output "figures/benchmark_cluster_xgboost_median.png" \
    --no-show

echo "======================================================================"
echo "Finished at `date`"
echo "Ergebnisse gespeichert in: results/benchmark_cluster_xgboost.csv"
echo "Grafik gespeichert in    : figures/benchmark_cluster_xgboost_median.png"
echo "======================================================================"
