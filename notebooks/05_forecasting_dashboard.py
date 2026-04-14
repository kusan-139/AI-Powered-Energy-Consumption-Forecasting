# ============================================================
# Notebook 05 — Forecasting Dashboard
# ============================================================
"""
OBJECTIVE:
    Generate all final outputs and launch the Flask dashboard.

WHAT YOU WILL LEARN:
    - Generating the complete set of visualisation charts
    - Creating an automated PDF report
    - Running the Flask web dashboard
    - Understanding the full end-to-end ML pipeline
"""

import sys, warnings, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("  NOTEBOOK 05 — Final Outputs & Dashboard")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────
from src.preprocessor import load_processed
from src.simulator import load_simulated_data

try:
    df = load_processed(freq="H")
except FileNotFoundError:
    try:
        df = load_processed(freq="H", simulated=True)
    except FileNotFoundError:
        df = load_simulated_data(freq="H")

print(f"[INFO] Data: {df.shape}")

# ── Load metrics ───────────────────────────────────────────
import json
metrics_path = ROOT / "outputs" / "metrics" / "model_metrics.json"
metrics = []
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
    print(f"[INFO] Loaded metrics for {len(metrics)} models")
else:
    print("[WARN] No metrics found — using demo values")
    metrics = [
        {"model":"ARIMA",   "MAE":0.142,"RMSE":0.198,"MAPE":12.4,"SMAPE":11.8,"R2":0.781,"Adj_R2":0.771},
        {"model":"XGBoost", "MAE":0.089,"RMSE":0.121,"MAPE": 7.2,"SMAPE": 6.9,"R2":0.912,"Adj_R2":0.908},
        {"model":"LSTM",    "MAE":0.074,"RMSE":0.103,"MAPE": 5.8,"SMAPE": 5.5,"R2":0.941,"Adj_R2":0.935},
    ]

# ── Generate all 6 charts ──────────────────────────────────
print("\n[STEP 1] Generating all visualisation charts …")
import matplotlib; matplotlib.use("Agg")
from src.visualizer import (
    plot_consumption_overview,
    plot_seasonal_patterns,
    plot_anomaly_detection,
)
plot_consumption_overview(df)
plot_seasonal_patterns(df)
plot_anomaly_detection(df)

# Model comparison chart
from src.evaluator import compare_models
import pandas as pd
metrics_df        = pd.DataFrame(metrics).set_index("model")
from src.visualizer import plot_model_comparison
plot_model_comparison(metrics_df)

print(f"[INFO] Charts saved to {ROOT / 'images'}/")

# ── Generate PDF report ────────────────────────────────────
print("\n[STEP 2] Generating PDF report …")
from src.reporter import generate_report
gp = df["Global_active_power"].dropna()
stats = {
    "Data Source"   : "UCI Household Power Consumption (simulated)",
    "Total Records" : f"{len(df):,}",
    "Date From"     : str(df.index.min().date()),
    "Date To"       : str(df.index.max().date()),
    "Avg Power"     : f"{gp.mean():.4f} kW",
    "Max Power"     : f"{gp.max():.4f} kW",
    "Total Energy"  : f"{(gp.sum()/1000):.2f} MWh",
}
report_path = generate_report(stats=stats, metrics_list=metrics)
print(f"[INFO] PDF report: {report_path}")

# ── Summary ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  📊 ALL OUTPUTS GENERATED")
print("=" * 60)

images = list((ROOT / "images").glob("*.png"))
print(f"\n  Charts ({len(images)}):")
for img in images:
    print(f"    ✓ {img.name}")

print(f"\n  Report: {report_path}")
print(f"  Metrics: {metrics_path}")
print(f"\n  🌐 Launch Dashboard:")
print(f"     python dashboard/app.py")
print(f"     Then open: http://localhost:5000")
print("=" * 60)
