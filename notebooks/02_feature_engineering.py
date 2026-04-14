# ============================================================
# Notebook 02 — Feature Engineering
# ============================================================
"""
OBJECTIVE:
    Transform raw energy data into ML-ready features.

WHAT YOU WILL LEARN:
    - Why feature engineering matters for time-series ML
    - Lag features (past values as predictors)
    - Rolling statistics (moving averages, rolling std)
    - Calendar encoding (hour, day, season)
    - Cyclical encoding (sin/cos transforms)
    - Interaction features (temp × hour, HDD/CDD)
    - Correlation analysis of engineered features
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMAGES_DIR = ROOT / "images"

print("=" * 60)
print("  NOTEBOOK 02 — Feature Engineering")
print("=" * 60)

# ── Load preprocessed data ──────────────────────────────────
from src.preprocessor import load_processed
from src.simulator import load_simulated_data

try:
    df = load_processed(freq="H")
except FileNotFoundError:
    try:
        df = load_processed(freq="H", simulated=True)
    except FileNotFoundError:
        df = load_simulated_data(freq="H")

print(f"\n[INFO] Input shape: {df.shape}")

# ── Run feature engineering ────────────────────────────────
from src.feature_engineer import engineer_features, get_feature_columns, train_test_split_temporal

df_feat = engineer_features(df, save=True)
feat_cols, target = get_feature_columns(df_feat)

print(f"\n[INFO] Feature count: {len(feat_cols)}")
print("\n[INFO] Feature groups:")
lag_feats     = [f for f in feat_cols if "lag" in f]
rolling_feats = [f for f in feat_cols if "roll" in f]
cal_feats     = [f for f in feat_cols if f in ["hour","dayofweek","month","season","is_weekend","is_peak_hour"]]
cyc_feats     = [f for f in feat_cols if "_sin" in f or "_cos" in f]
print(f"  Lag features        : {len(lag_feats)}    {lag_feats[:5]} …")
print(f"  Rolling features    : {len(rolling_feats)}")
print(f"  Calendar features   : {len(cal_feats)}")
print(f"  Cyclical encodings  : {len(cyc_feats)}")

# ── Feature correlation heatmap ────────────────────────────
print("\n[CHART] Feature correlation with target …")
top_feats = [target] + feat_cols[:20]
corr = df_feat[top_feats].corr()[target].drop(target).sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0D1117")
ax.set_facecolor("#0D1117")
colors = ["#ff4d6d" if v < 0 else "#00c2ff" for v in corr.values]
bars = ax.barh(corr.index, corr.values, color=colors, alpha=0.8)
ax.set_title("Feature Correlation with Target (Global_active_power)", color="#e6edf3", fontsize=12)
ax.set_xlabel("Pearson Correlation", color="#8b949e")
ax.tick_params(colors="#8b949e")
ax.axvline(0, color="#21262D", linewidth=1)
ax.grid(axis="x", color="#21262D", alpha=0.5)
plt.tight_layout()
fig.savefig(IMAGES_DIR / "feature_correlation.png", dpi=120, bbox_inches="tight", facecolor="#0D1117")
plt.close(fig)
print(f"  Saved → {IMAGES_DIR / 'feature_correlation.png'}")

# ── Train/test split ───────────────────────────────────────
train_df, test_df = train_test_split_temporal(df_feat)
print(f"\n[SPLIT] Train: {len(train_df):,}  |  Test: {len(test_df):,}")

# ── Lag analysis: Auto-correlation table ───────────────────
print("\n[INFO] Lag correlation table:")
lags = [1, 2, 6, 12, 24, 48, 168]
print(f"  {'Lag':>6}  {'Correlation':>12}")
for lag in lags:
    corr_val = df_feat[target].autocorr(lag=lag)
    print(f"  {lag:>6}h  {corr_val:>12.4f}")

print("\n✅ NOTEBOOK 02 COMPLETE")
