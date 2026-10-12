#!/usr/bin/bash
# ==============================================================================
# Slurm Batch-Skript für 20-Run Thermal Throttling Stresstest (HPC SoSe 2026)
# Führt 20 aufeinanderfolgende Trainingsläufe unter 100% Volllast auf allen
# 24 Kernen aus, um die thermische und frequenzielle Stabilität der Xeon-CPUs
# im Vergleich zum thermisch drosselnden Mac M4 Laptop zu beweisen.
# ==============================================================================
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION>

#SBATCH --job-name=cluster_throttling
#SBATCH --output=%u.log.%j.out
#SBATCH --error=%u.log.%j.err

#SBATCH --time=0-00:45:00
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

# 4. 20 konsekutive Durchläufe unter 24-Kern-Volllast
echo "[3/3] Starte 20-Run Throttling Stresstest auf 24 Kernen..."
python3 scripts/benchmark_throttling.py \
    --data "$DATASET_PATH" \
    --runs 20 \
    --n-jobs 24 \
    --label "Intel_Xeon_Gold_6226_Cluster" \
    --output "results/cluster_throttling_20runs.csv" \
    --plot "figures/cluster_thermal_throttling_20runs.png"

echo "======================================================================"
echo "Finished at `date`"
echo "CSV exportiert nach : results/cluster_throttling_20runs.csv"
echo "Plot exportiert nach: figures/cluster_thermal_throttling_20runs.png"
echo "======================================================================"
