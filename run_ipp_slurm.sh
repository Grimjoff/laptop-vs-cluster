#!/usr/bin/bash
# ==============================================================================
# Slurm Batch-Skript für IPyParallel Strong-Scaling Benchmark (HPC SoSe 2026)
# Adjust account/partition below for your cluster
# ==============================================================================
#SBATCH --account=<YOUR_ACCOUNT>   # set to your cluster account
#SBATCH --partition=<YOUR_PARTITION>

#SBATCH --job-name=ipp_benchmark
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

# 2. Module laden wie in Vorlesung & Übung
echo "[1/6] Lade Module und aktiviere Python-Environment..."
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
python3 -c "import ipyparallel, sklearn, xgboost; print('Python-Pakete erfolgreich verifiziert.')"

# 3. Prozess-Threading für IPyParallel auf 1 setzen (verhindert Überbelegung)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 4. Datensatz prüfen
DATASET_PATH="data/delays_zurich_transport_45034.parquet"
if [ ! -f "$DATASET_PATH" ]; then
    echo "[2/6] Datensatz nicht gefunden. Starte Download..."
    python3 scripts/download_45034.py
fi

# 5. Alte Stale-PIDs und Zombie-Sockets von vorherigen Jobs bereinigen
echo "[2b/6] Bereinige alte IPyParallel Profile, PIDs und Sockets..."
ipcluster clean --profile=default 2>/dev/null || true
pkill -u $USER -f ipengine 2>/dev/null || true
pkill -u $USER -f ipcontroller 2>/dev/null || true
rm -rf ~/.ipython/profile_default/pid/* 2>/dev/null || true
rm -rf ~/.ipython/profile_default/security/* 2>/dev/null || true
sleep 3

# 6. IPyParallel Cluster frisch starten (1 Controller + 24 Engines)
echo "[3/6] Starte IPyParallel Cluster frisch (1 Controller + 24 Engines)..."
ipcluster start -n 24 --daemonize
sleep 10

# 6. Prüfen, ob Engines online sind
python3 -c "import ipyparallel as ipp; rc = ipp.Client(timeout=10); print(f'Erfolgreich verbunden mit {len(rc.ids)} Engines: {rc.ids}')"

# 7. Strong Scaling Benchmark ausführen (500.000 Samples für sauberen 1..24 Kern-Sweep)
export PYTHONUNBUFFERED=1
echo "[4/6] Starte Strong Scaling Benchmark (1, 2, 4, 8, 12, 16, 24 Kerne)..."
python3 scripts/benchmark_strong_scaling.py \
    --data "$DATASET_PATH" \
    --output "results/benchmark_strong_scaling.csv" \
    --sample-size 500000 \
    --cores 1 2 4 8 12 16 24 \
    --runs 2

# Optional: Paralleles Ensemble (Tag 8)
echo "[5/6] Starte Parallelen Ensemble-Benchmark..."
python3 scripts/benchmark_ensemble.py \
    --data "$DATASET_PATH" \
    --sample-size 500000 \
    --output "results/benchmark_ensemble_results.csv" || true

# 8. Veröffentlichungs-Grafiken (Speedup & cc-NUMA) generieren
echo "[6/6] Generiere Veröffentlichungs-Grafiken (Speedup & cc-NUMA Effizienz)..."
python3 scripts/plot_scaling_and_numa.py \
    --csv "results/benchmark_strong_scaling.csv" \
    --output-dir "figures" \
    --no-show

# 9. Aufräumen & Beenden
echo "Beende IPyParallel Controller & Engines..."
ipcluster stop || true
pkill -u $USER -f ipengine || true
pkill -u $USER -f ipcontroller || true
kill -9 $CONTROLLER_PID 2>/dev/null || true

echo "Finished at `date`"
