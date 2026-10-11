import os
import sys
import time
import json
import re
import glob
import shutil
import argparse
import platform
import threading
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def probe_cpu_temperature():
    """
    Cross-platform CPU temperature probe.
    Supports macOS (macmon for Apple Silicon / osx-cpu-temp / powermetrics),
    Linux (Cluster / sysfs / psutil), and fallback.
    Returns float temperature in Celsius or None.
    """
    # 1. macOS (Apple Silicon via macmon or fallback)
    if platform.system() == "Darwin":
        # Check macmon (Apple Silicon sudoless telemetry)
        if shutil.which("macmon"):
            try:
                proc = subprocess.Popen(
                    ["macmon", "pipe", "-i", "100"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                line = proc.stdout.readline()
                proc.terminate()
                proc.wait(timeout=1.0)
                if line:
                    data = json.loads(line)
                    t = data.get("temp", {}).get("cpu_temp_avg")
                    if t and float(t) > 5.0:
                        return float(t)
            except Exception:
                pass

        # Check osx-cpu-temp (legacy Intel Mac Homebrew tool)
        if shutil.which("osx-cpu-temp"):
            try:
                res = subprocess.run(["osx-cpu-temp"], capture_output=True, text=True, timeout=1)
                match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*°?C", res.stdout)
                if match:
                    val = float(match.group(1))
                    if val > 5.0:  # Ignore 0.0 °C dummy readings on Apple Silicon
                        return val
            except Exception:
                pass

        # Check powermetrics (built into macOS, requires sudo)
        try:
            cmd = ["powermetrics", "-n", "1", "-i", "200", "--samplers", "smc"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5)
            match = re.search(r"CPU die temperature:\s*([0-9.]+)\s*C", res.stdout)
            if match:
                val = float(match.group(1))
                if val > 5.0:
                    return val
        except Exception:
            pass

    # 2. Check psutil.sensors_temperatures (Standard on Linux, some Windows)
    if HAS_PSUTIL and hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name in ["coretemp", "cpu_thermal", "k10temp", "acpitz", "cpu-thermal"]:
                    if name in temps and len(temps[name]) > 0:
                        vals = [entry.current for entry in temps[name] if entry.current is not None and entry.current > 5.0]
                        if vals:
                            return float(sum(vals) / len(vals))
                # Fallback to any sensor
                for entries in temps.values():
                    vals = [entry.current for entry in entries if entry.current is not None and entry.current > 5.0]
                    if vals:
                        return float(sum(vals) / len(vals))
        except Exception:
            pass

    # 3. Linux SysFS (/sys/class/thermal/thermal_zone*/temp)
    if platform.system() == "Linux":
        try:
            zone_files = glob.glob("/sys/class/thermal/thermal_zone*/temp")
            readings = []
            for zf in zone_files:
                with open(zf, "r") as f:
                    val = float(f.read().strip())
                    if val > 1000:
                        val /= 1000.0  # milli-degrees to degrees
                    if 15.0 <= val <= 115.0:
                        readings.append(val)
            if readings:
                return float(sum(readings) / len(readings))
        except Exception:
            pass

    return None


class HardwareSampler(threading.Thread):
    """
    Background sampling thread that tracks CPU Temperature, Clock Frequency, and Utilization
    during each model training execution.
    Supports real-time streaming telemetry via macmon on Apple Silicon and psutil on Linux.
    """
    def __init__(self, interval_sec=1.0):
        super().__init__(daemon=True)
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        init_t = probe_cpu_temperature()
        self.temps = [init_t] if (init_t and init_t > 5.0) else []
        self.clocks = []
        self.cpu_pcts = []
        self.macmon_proc = None

    def run(self):
        # Path A: Apple Silicon with macmon stream
        if platform.system() == "Darwin" and shutil.which("macmon"):
            try:
                interval_ms = max(200, int(self.interval_sec * 1000))
                self.macmon_proc = subprocess.Popen(
                    ["macmon", "pipe", "-i", str(interval_ms)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                while not self.stop_event.is_set():
                    line = self.macmon_proc.stdout.readline()
                    if not line:
                        break
                    try:
                        data = json.loads(line)
                        t = data.get("temp", {}).get("cpu_temp_avg")
                        if t and float(t) > 5.0:
                            self.temps.append(float(t))

                        # P-Core frequency in MHz
                        freq = data.get("pcpu_freq_mhz")
                        if not freq:
                            cores = data.get("pcpu_cores", []) + data.get("ecpu_cores", [])
                            if cores:
                                freq = sum(c.get("freq_mhz", 0) for c in cores) / len(cores)
                        if freq and float(freq) > 100:  # Avoid dummy 4 MHz values
                            self.clocks.append(float(freq))

                        usage = data.get("cpu_usage_pct") or data.get("cpu_active_ratio") or data.get("cpu_scaled_ratio")
                        if usage is not None:
                            pct_val = float(usage) * 100.0 if float(usage) <= 1.0 else float(usage)
                            self.cpu_pcts.append(pct_val)
                    except Exception:
                        pass
                return
            except Exception:
                pass

        # Path B: Standard polling loop (Linux / Cluster / Fallback)
        while not self.stop_event.is_set():
            temp = probe_cpu_temperature()
            if temp is not None and temp > 5.0:
                self.temps.append(temp)

            if HAS_PSUTIL:
                try:
                    freq = psutil.cpu_freq()
                    if freq and freq.current and freq.current > 100:  # Filter out 4 MHz junk
                        self.clocks.append(freq.current)
                except Exception:
                    pass

                try:
                    pct = psutil.cpu_percent(interval=None)
                    if pct:
                        self.cpu_pcts.append(pct)
                except Exception:
                    pass

            self.stop_event.wait(self.interval_sec)

    def stop_and_aggregate(self):
        self.stop_event.set()
        if self.macmon_proc:
            try:
                self.macmon_proc.terminate()
                self.macmon_proc.wait(timeout=1.0)
            except Exception:
                pass

        self.join(timeout=2.0)

        avg_temp = round(sum(self.temps) / len(self.temps), 1) if self.temps else None
        max_temp = round(max(self.temps), 1) if self.temps else None
        avg_clock = round(sum(self.clocks) / len(self.clocks), 0) if self.clocks else None
        avg_cpu = round(sum(self.cpu_pcts) / len(self.cpu_pcts), 1) if self.cpu_pcts else None

        return {
            "avg_temp_c": avg_temp,
            "max_temp_c": max_temp,
            "avg_clock_mhz": avg_clock,
            "avg_cpu_pct": avg_cpu,
        }


def run_throttling_benchmark(
    parquet_path="data/delays_zurich_transport_45034.parquet",
    csv_file="throttling_results.csv",
    plot_path="figures/thermal_throttling_runs.png",
    total_runs=15,
    model_type="rf",
    sample_size=None,
    n_jobs=-1,
    machine_label=None,
    n_estimators=100,
    max_depth=15,
    random_state=42,
    show_plot=False,
):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

    if machine_label is None:
        machine_label = f"{platform.system()}_{platform.machine()}"

    max_cores = os.cpu_count() or 1
    active_cores = max_cores if (n_jobs == -1 or n_jobs is None) else min(n_jobs, max_cores)

    # Initial hardware check
    initial_temp = probe_cpu_temperature()
    temp_support = f"Available ({initial_temp:.1f} °C)" if initial_temp is not None else "Sensor not directly exposed (logging clock/CPU usage)"

    print("=" * 80)
    print("        HPC THERMAL THROTTLING BENCHMARK (Continuous Load & Telemetry)        ")
    print("=" * 80)
    print(f"Machine Label      : {machine_label}")
    print(f"Platform           : {platform.platform()}")
    print(f"Detected CPU Cores : {max_cores} cores")
    print(f"Active Worker Cores: {active_cores} cores (n_jobs={n_jobs})")
    print(f"Temperature Sensor : {temp_support}")
    print(f"Model Architecture : {model_type.upper()}")
    print(f"Total Iterations   : {total_runs} consecutive runs (ZERO delay)")
    print("=" * 80)

    # 1. Load dataset
    print(f"\nLoading dataset: {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)
    print(f"Loaded {total_rows:,} rows and {len(df.columns)} columns.")

    if sample_size and sample_size < total_rows:
        print(f"Using a sample of {sample_size:,} rows for benchmark...")
        data = df.sample(n=sample_size, random_state=random_state)
    else:
        data = df

    # 2. Features and split
    target_col = "delay"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    print(f"Training set: {len(X_train):,} samples\n")

    os.makedirs(os.path.dirname(csv_file) or ".", exist_ok=True)
    if plot_path:
        os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)

    results = []
    baseline_time = None

    print("-" * 80)
    print(f"{'Run':<5} | {'Duration (s)':<13} | {'vs Run 1':<10} | {'Avg Temp':<10} | {'Max Temp':<10} | {'Avg Clock':<11} | {'Timestamp':<19}")
    print("-" * 80)

    for run_idx in range(1, total_runs + 1):
        if model_type.lower() == "rf":
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=active_cores,
                random_state=random_state,
            )
        elif model_type.lower() == "xgb":
            if not HAS_XGB:
                raise ImportError("xgboost is not installed. Run `pip install xgboost` or use --model rf")
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                tree_method="hist",
                n_jobs=active_cores,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'rf' or 'xgb'.")

        # Start hardware monitoring thread during fit
        sampler = HardwareSampler(interval_sec=1.0)
        sampler.start()

        timestamp_start = time.strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.time()
        model.fit(X_train, y_train)
        duration = round(time.time() - t0, 4)

        # Stop telemetry sampler
        telemetry = sampler.stop_and_aggregate()

        if baseline_time is None:
            baseline_time = duration
            delta_str = "Baseline"
            slowdown_pct = 0.0
        else:
            slowdown_pct = round(((duration - baseline_time) / baseline_time) * 100, 2)
            sign = "+" if slowdown_pct >= 0 else ""
            delta_str = f"{sign}{slowdown_pct:.1f}%"

        temp_str = f"{telemetry['avg_temp_c']} °C" if telemetry['avg_temp_c'] is not None else "N/A"
        max_temp_str = f"{telemetry['max_temp_c']} °C" if telemetry['max_temp_c'] is not None else "N/A"
        clock_str = f"{int(telemetry['avg_clock_mhz'])} MHz" if telemetry['avg_clock_mhz'] is not None else "N/A"

        print(f"Run {run_idx:<2}| {duration:9.2f}s    | {delta_str:<10} | {temp_str:<10} | {max_temp_str:<10} | {clock_str:<11} | {timestamp_start}")

        record = {
            "machine": machine_label,
            "model": model_type.upper(),
            "n_jobs": active_cores,
            "sample_size": len(data),
            "run_index": run_idx,
            "duration_sec": duration,
            "slowdown_pct": slowdown_pct,
            "avg_temp_c": telemetry["avg_temp_c"],
            "max_temp_c": telemetry["max_temp_c"],
            "avg_clock_mhz": telemetry["avg_clock_mhz"],
            "avg_cpu_pct": telemetry["avg_cpu_pct"],
            "timestamp": timestamp_start,
        }
        results.append(record)

        # Append row immediately to CSV
        row_df = pd.DataFrame([record])
        file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
        row_df.to_csv(csv_file, mode="a", header=not file_exists, index=False)

    df_results = pd.DataFrame(results)

    print("-" * 80)
    print(f"\nAll {total_runs} runs completed successfully.")
    print(f"Results appended to: {csv_file}")

    first_run = df_results.iloc[0]["duration_sec"]
    last_run = df_results.iloc[-1]["duration_sec"]
    overall_slowdown = ((last_run - first_run) / first_run) * 100
    print(f"\nSummary:")
    print(f"  Run 1  (Cold baseline) : {first_run:.2f}s")
    print(f"  Run {total_runs} (Heat-soaked)   : {last_run:.2f}s")
    print(f"  Total Thermal Slowdown : {overall_slowdown:+.1f}%\n")

    # 3. Generate Plot
    if plot_path:
        _plot_throttling(df_results, plot_path, machine_label, show_plot)


def _plot_throttling(df_results, plot_path, machine_label, show_plot):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    has_temp = df_results["avg_temp_c"].notnull().any()
    runs = df_results["run_index"]
    durations = df_results["duration_sec"]

    if has_temp:
        # Create dual-panel stacked plot (Top: Duration, Bottom: Temperature & Clock)
        fig, (ax_dur, ax_temp) = plt.subplots(2, 1, figsize=(9.5, 8.5), dpi=150, sharex=True)

        # Top Plot: Execution Time
        ax_dur.plot(runs, durations, marker="o", linewidth=2.2, markersize=7, color="#e11d48", label="Execution Duration", zorder=3)
        r1_val = durations.iloc[0]
        r_end_val = durations.iloc[-1]
        ax_dur.annotate(f"Run 1: {r1_val:.1f}s\n(Cold)", xy=(runs.iloc[0], r1_val), xytext=(15, 10), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1), fontweight="bold", fontsize=9, color="#1e293b")
        diff_pct = ((r_end_val - r1_val) / r1_val) * 100
        sign = "+" if diff_pct >= 0 else ""
        ax_dur.annotate(f"Run {runs.iloc[-1]}: {r_end_val:.1f}s\n({sign}{diff_pct:.1f}%)", xy=(runs.iloc[-1], r_end_val), xytext=(-55, 10),
                        textcoords="offset points", arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1), fontweight="bold", fontsize=9, color="#9f1239")
        ax_dur.set_title(f"Thermal Throttling & Telemetry: {machine_label} (Sustained Workload)", fontsize=13, fontweight="bold", pad=12)
        ax_dur.set_ylabel("Execution Time (seconds)", fontsize=11, labelpad=8)
        ax_dur.grid(True, linestyle="--", alpha=0.6)
        ax_dur.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")
        ax_dur.set_ylim(bottom=max(0, durations.min() * 0.9), top=durations.max() * 1.15)

        # Bottom Plot: Temperature
        ax_temp.plot(runs, df_results["avg_temp_c"], marker="s", linewidth=2.0, markersize=6, color="#f97316", label="Avg CPU Temperature (°C)", zorder=3)
        if df_results["max_temp_c"].notnull().any():
            ax_temp.plot(runs, df_results["max_temp_c"], marker="^", linestyle=":", linewidth=1.5, markersize=6, color="#dc2626", label="Peak CPU Temp (°C)", zorder=2)
        ax_temp.set_xlabel("Consecutive Run Index (1 to 15)", fontsize=11, labelpad=10)
        ax_temp.set_ylabel("Temperature (°C)", fontsize=11, labelpad=8)
        ax_temp.set_xticks(runs.tolist())
        ax_temp.grid(True, linestyle="--", alpha=0.6)
        ax_temp.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1")

        fig.tight_layout()
    else:
        # Single panel plot if temperature sensor is unavailable
        fig, ax_dur = plt.subplots(figsize=(9, 5.5), dpi=150)
        ax_dur.plot(runs, durations, marker="o", linewidth=2.2, markersize=7, color="#e11d48", label=f"{machine_label} (Sustained Load)", zorder=3)
        r1_val = durations.iloc[0]
        r_end_val = durations.iloc[-1]
        ax_dur.annotate(f"Run 1: {r1_val:.1f}s\n(Cold)", xy=(runs.iloc[0], r1_val), xytext=(15, 15), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1), fontweight="bold", fontsize=9, color="#1e293b")
        diff_pct = ((r_end_val - r1_val) / r1_val) * 100
        sign = "+" if diff_pct >= 0 else ""
        ax_dur.annotate(f"Run {runs.iloc[-1]}: {r_end_val:.1f}s\n({sign}{diff_pct:.1f}%)", xy=(runs.iloc[-1], r_end_val), xytext=(-45, 15),
                        textcoords="offset points", arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1), fontweight="bold", fontsize=9, color="#9f1239")
        ax_dur.set_title(f"Thermal Throttling: Execution Time vs. Iterations ({machine_label})", fontsize=13, fontweight="bold", pad=15)
        ax_dur.set_xlabel("Consecutive Run Index (1 to 15)", fontsize=11, labelpad=10)
        ax_dur.set_ylabel("Execution Time (seconds)", fontsize=11, labelpad=10)
        ax_dur.set_xticks(runs.tolist())
        ax_dur.grid(True, linestyle="--", alpha=0.6)
        ax_dur.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
        ax_dur.set_ylim(bottom=max(0, durations.min() * 0.9), top=durations.max() * 1.15)
        fig.tight_layout()

    fig.savefig(plot_path, dpi=300)
    print(f"Throttling plot saved to: {plot_path}")

    if show_plot:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thermal Throttling Benchmark under continuous load with telemetry.")
    parser.add_argument("--data", default="data/delays_zurich_transport_45034.parquet", help="Path to parquet file")
    parser.add_argument("--output", default="throttling_results.csv", help="CSV output path")
    parser.add_argument("--plot", default="figures/thermal_throttling_runs.png", help="Figure output path")
    parser.add_argument("--runs", type=int, default=15, help="Number of consecutive iterations (default: 15)")
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf", help="Model to train (rf: RandomForest, xgb: XGBoost)")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size (None for full 5.46M dataset)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of CPU cores to utilize (-1 for all)")
    parser.add_argument("--label", default=None, help="Machine label for reporting (e.g. MacBook_Air_M4, Cluster)")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=15, help="Tree max depth")
    parser.add_argument("--show", action="store_true", help="Display plot window upon completion")
    args = parser.parse_args()

    run_throttling_benchmark(
        parquet_path=args.data,
        csv_file=args.output,
        plot_path=args.plot,
        total_runs=args.runs,
        model_type=args.model,
        sample_size=args.sample_size,
        n_jobs=args.n_jobs,
        machine_label=args.label,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        show_plot=args.show,
    )
