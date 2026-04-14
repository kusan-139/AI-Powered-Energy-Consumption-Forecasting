# ============================================================
# Notebook 01 — Data Exploration & EDA
# Run: python notebooks/01_data_exploration.py
# Or:  jupyter notebook notebooks/01_data_exploration.ipynb
#      (after running: python notebooks/generate_notebooks.py)
# ============================================================

"""
OBJECTIVE:
    Explore the UCI Household Power Consumption dataset (or simulated data).
    Understand its structure, distributions, and seasonality patterns.

WHAT YOU WILL LEARN:
    - How to load and inspect large time-series datasets
    - Missing value analysis and imputation strategies
    - Distribution analysis (histograms, KDE)
    - Temporal decomposition (trend, seasonality, residuals)
    - Autocorrelation and partial autocorrelation plots
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor"  : "#0D1117",
    "axes.facecolor"    : "#0D1117",
    "axes.edgecolor"    : "#21262D",
    "axes.labelcolor"   : "#E6EDF3",
    "xtick.color"       : "#8b949e",
    "ytick.color"       : "#8b949e",
    "text.color"        : "#E6EDF3",
    "grid.color"        : "#21262D",
    "grid.alpha"        : 0.5,
    "font.family"       : "DejaVu Sans",
})

print("=" * 60)
print("  NOTEBOOK 01 — Data Exploration & EDA")
print("=" * 60)

# ── STEP 1: Load data ──────────────────────────────────────────
print("\n[STEP 1] Loading data …")

try:
    from src.preprocessor import load_processed
    df = load_processed(freq="H", simulated=False)
    print("[INFO] Loaded UCI processed data")
except FileNotFoundError:
    try:
        from src.preprocessor import load_processed
        df = load_processed(freq="H", simulated=True)
        print("[INFO] Loaded simulated data")
    except FileNotFoundError:
        from src.simulator import generate_synthetic_data
        from src.preprocessor import preprocess_simulated
        print("[INFO] Generating synthetic data …")
        df = generate_synthetic_data(freq="H", save=True)
        df = preprocess_simulated(df)

print(f"\n  Shape      : {df.shape}")
print(f"  Columns    : {list(df.columns)}")
print(f"  Date range : {df.index.min()} → {df.index.max()}")
print(f"  Duration   : {(df.index.max() - df.index.min()).days} days")

# ── STEP 2: Basic statistics ───────────────────────────────────
print("\n[STEP 2] Descriptive Statistics —")
print(df.describe().round(4).to_string())

# ── STEP 3: Missing value analysis ────────────────────────────
print("\n[STEP 3] Missing Values —")
missing = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)
missing_df = pd.DataFrame({"Missing": missing, "Pct (%)": missing_pct})
print(missing_df.to_string())

# ── STEP 4: Full time-series plot ─────────────────────────────
print("\n[STEP 4] Generating consumption overview chart …")
from src.visualizer import plot_consumption_overview
plot_consumption_overview(df)

# ── STEP 5: Seasonal patterns heatmap ─────────────────────────
print("[STEP 5] Generating seasonal patterns heatmap …")
from src.visualizer import plot_seasonal_patterns
plot_seasonal_patterns(df)

# ── STEP 6: Distribution analysis ─────────────────────────────
print("[STEP 6] Distribution analysis …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
col = "Global_active_power"

axes[0].hist(df[col].dropna(), bins=80, color="#00c2ff", alpha=0.7, edgecolor="#0D1117")
axes[0].set_title("Power Distribution (Histogram)", fontsize=12)
axes[0].set_xlabel("Active Power (kW)")
axes[0].set_ylabel("Frequency")

df[col].dropna().plot(kind="kde", ax=axes[1], color="#00f5d4", linewidth=2)
axes[1].set_title("Power Distribution (KDE)", fontsize=12)
axes[1].set_xlabel("Active Power (kW)")

plt.tight_layout()
fig.savefig(IMAGES_DIR / "distribution_analysis.png", dpi=120, bbox_inches="tight",
            facecolor="#0D1117")
plt.close(fig)
print(f"  Saved → {IMAGES_DIR / 'distribution_analysis.png'}")

# ── STEP 7: Time decomposition ────────────────────────────────
print("[STEP 7] Time series decomposition …")
daily = df[col].resample("D").mean().dropna()
# Use last 365 days for decomposition
daily_slice = daily.iloc[-365:]

try:
    decomp = seasonal_decompose(daily_slice, model="additive", period=7)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    for ax, component, name, color in zip(
        axes,
        [daily_slice, decomp.trend, decomp.seasonal, decomp.resid],
        ["Observed", "Trend", "Seasonality (Weekly)", "Residuals"],
        ["#00c2ff", "#00f5d4", "#bf5af2", "#ff4d6d"],
    ):
        ax.plot(component, color=color, linewidth=1.2 if name != "Observed" else 0.8)
        ax.set_title(name, fontsize=11)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Time Series Decomposition (Additive)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "decomposition.png", dpi=120, bbox_inches="tight",
                facecolor="#0D1117")
    plt.close(fig)
    print(f"  Saved → {IMAGES_DIR / 'decomposition.png'}")
except Exception as e:
    print(f"  [WARN] Decomposition skipped: {e}")

# ── STEP 8: ACF / PACF ────────────────────────────────────────
print("[STEP 8] ACF / PACF analysis …")
hourly_sample = df[col].dropna().iloc[-1000:]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(hourly_sample,  lags=48, ax=ax1, color="#00c2ff")
ax1.set_title("Autocorrelation (ACF)", fontsize=11)
plot_pacf(hourly_sample, lags=48, ax=ax2, color="#00f5d4")
ax2.set_title("Partial Autocorrelation (PACF)", fontsize=11)
plt.tight_layout()
fig.savefig(IMAGES_DIR / "acf_pacf.png", dpi=120, bbox_inches="tight",
            facecolor="#0D1117")
plt.close(fig)
print(f"  Saved → {IMAGES_DIR / 'acf_pacf.png'}")

print("\n" + "=" * 60)
print("  ✅ NOTEBOOK 01 COMPLETE")
print(f"  All charts saved to {IMAGES_DIR}")
print("=" * 60)
