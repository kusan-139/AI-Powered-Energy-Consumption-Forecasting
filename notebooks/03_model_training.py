# ============================================================
# Notebook 03 — Model Training
# ============================================================
"""
OBJECTIVE:
    Train all three forecasting models step-by-step.

WHAT YOU WILL LEARN:
    - How to train ARIMA / SARIMA on time-series data
    - How to train XGBoost with lag features
    - How to build and train an LSTM neural network
    - Saving trained models for deployment
    - Interpreting training output and loss curves
"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMAGES_DIR = ROOT / "images"

print("=" * 60)
print("  NOTEBOOK 03 — Model Training")
print("=" * 60)

# ── Load features ──────────────────────────────────────────
from src.preprocessor import load_processed
from src.simulator import load_simulated_data
from src.feature_engineer import engineer_features, get_feature_columns, train_test_split_temporal

try:
    df_raw = load_processed(freq="H")
except FileNotFoundError:
    try:
        df_raw = load_processed(freq="H", simulated=True)
    except FileNotFoundError:
        df_raw = load_simulated_data(freq="H")

df_feat = engineer_features(df_raw, save=False)
feat_cols, target = get_feature_columns(df_feat)
train_df, test_df = train_test_split_temporal(df_feat)

X_train = train_df[feat_cols]
y_train = train_df[target]
X_test  = test_df[feat_cols]
y_test  = test_df[target]

# ── MODEL 1: ARIMA ─────────────────────────────────────────
print("\n━━━ MODEL 1: ARIMA ━━━")
from src.models.arima_model import ARIMAForecaster

series = df_raw["Global_active_power"].dropna().iloc[-2000:]
split  = int(len(series) * 0.8)
arima_train, arima_test = series.iloc[:split], series.iloc[split:]

arima = ARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,24))
arima.fit(arima_train)
arima_preds = arima.predict(steps=len(arima_test))
arima.save()
print(f"  Predicted {len(arima_preds)} steps")

# ── MODEL 2: XGBOOST ───────────────────────────────────────
print("\n━━━ MODEL 2: XGBOOST ━━━")
from src.models.xgboost_model import XGBoostForecaster

n_val = int(len(X_train) * 0.1)
X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

xgb = XGBoostForecaster()
xgb.fit(X_tr, y_tr, X_val, y_val)
xgb_preds  = xgb.predict(X_test)
importance = xgb.get_feature_importance()
xgb.save()

# Feature importance chart
from src.visualizer import plot_feature_importance
plot_feature_importance(importance)

# ── MODEL 3: LSTM ──────────────────────────────────────────
print("\n━━━ MODEL 3: LSTM ━━━")
from src.models.lstm_model import LSTMForecaster
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
series_all  = df_raw["Global_active_power"].dropna()
series_norm = scaler.fit_transform(series_all.values.reshape(-1,1)).flatten()

lstm = LSTMForecaster(look_back=24, epochs=15)
lstm.fit(series_norm)
lstm.save()

# Plot training history
hist = lstm.get_training_history()
if hist:
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    ax.plot(hist["loss"],     color="#00c2ff", linewidth=2, label="Train Loss")
    ax.plot(hist["val_loss"], color="#ff4d6d", linewidth=2, label="Val Loss", linestyle="--")
    ax.set_title("LSTM Training History", color="#e6edf3")
    ax.set_xlabel("Epoch", color="#8b949e")
    ax.set_ylabel("Huber Loss", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.grid(color="#21262D", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "lstm_training_history.png", dpi=120, bbox_inches="tight", facecolor="#0D1117")
    plt.close(fig)
    print(f"  Training history saved → {IMAGES_DIR / 'lstm_training_history.png'}")

print("\n✅ NOTEBOOK 03 COMPLETE — all 3 models trained and saved")
