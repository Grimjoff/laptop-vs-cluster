#!/usr/bin/bash
# ==============================================================================
# Slurm Batch-Skript für Intel Extension for Scikit-Learn (HPC SoSe 2026)
# Vergleicht Stock Scikit-Learn gegen Intel oneDAL AVX-512 auf Intel Xeon Gold
# ==============================================================================
#SBATCH --account=<YOUR_ACCOUNT>   # set to your cluster account
#SBATCH --partition=<YOUR_PARTITION>

#SBATCH --job-name=intelex_benchmark
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

# 3. Sicherstellen, dass scikit-learn-intelex auf dem Xeon-Knoten installiert ist
echo "[2/4] Prüfe scikit-learn-intelex Installation..."
python3 -c "import sklearnex; print(f'sklearnex Version: {sklearnex.__version__}')" 2>/dev/null || {
    echo "Installiere scikit-learn-intelex für x86_64 / Intel Xeon..."
    pip install scikit-learn-intelex
}

# 4. Datensatz prüfen
DATASET_PATH="data/delays_zurich_transport_45034.parquet"
if [ ! -f "$DATASET_PATH" ]; then
    echo "[3/4] Datensatz nicht gefunden. Starte Download..."
    python3 scripts/download_45034.py
fi

# 5. Benchmark ausführen: Stock vs. Intel Extension auf 24 Kernen (3 Wiederholungen)
echo "[4/4] Starte Benchmark: Stock Scikit-Learn vs. Intel Extension (24 Kerne)..."
python3 scripts/benchmark_intelex.py \
    --data "$DATASET_PATH" \
    --output "results/benchmark_intelex.csv" \
    --plot-path "figures/benchmark_stock_vs_intelex.png" \
    --cores 24 \
    --runs 3

echo "======================================================================"
echo "Finished at `date`"
echo "Ergebnisse gespeichert in: results/benchmark_intelex.csv"
echo "Grafik gespeichert in    : figures/benchmark_stock_vs_intelex.png"
echo "======================================================================"
