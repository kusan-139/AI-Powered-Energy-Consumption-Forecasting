"""
visualizer.py
=============
Generates all 6 publication-quality charts for the project.

Chart inventory:
  1. consumption_overview.png   — Full time-series + 7-day moving average
  2. seasonal_patterns.png      — Hour × Day-of-week heatmap
  3. feature_importance.png     — XGBoost top-15 features
  4. model_comparison.png       — MAE/RMSE/MAPE bar chart (all 3 models)
  5. lstm_forecast.png          — LSTM actual vs. predicted + shaded interval
  6. anomaly_detection.png      — Z-score anomaly overlay on time-series
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style ──────────────────────────────
PALETTE   = ["#00C2FF", "#FF6B6B", "#A8FF78", "#FFD93D", "#C77DFF"]
BG_COLOR  = "#0D1117"
GRID_COLOR= "#21262D"
TEXT_COLOR = "#E6EDF3"

plt.rcParams.update({
    "figure.facecolor"  : BG_COLOR,
    "axes.facecolor"    : BG_COLOR,
    "axes.edgecolor"    : GRID_COLOR,
    "axes.labelcolor"   : TEXT_COLOR,
    "axes.titlecolor"   : TEXT_COLOR,
    "xtick.color"       : TEXT_COLOR,
    "ytick.color"       : TEXT_COLOR,
    "text.color"        : TEXT_COLOR,
    "grid.color"        : GRID_COLOR,
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.4,
    "legend.facecolor"  : "#161B22",
    "legend.edgecolor"  : GRID_COLOR,
    "font.family"       : "DejaVu Sans",
    "font.size"         : 11,
})

TARGET = "Global_active_power"


def _save(fig, name: str) -> Path:
    path = IMAGES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path


# ──────────────────────────────────────────────
# 1. CONSUMPTION OVERVIEW
# ──────────────────────────────────────────────

def plot_consumption_overview(df: pd.DataFrame,
                              col: str = TARGET) -> Path:
    """Full time-series line chart with 7-day moving average."""
    print("\n[VIZ] Plotting consumption overview …")
    series = df[col].resample("D").mean().dropna()
    ma7    = series.rolling(7, center=True).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series.index, series.values, color=PALETTE[0],
            linewidth=0.7, alpha=0.6, label="Daily Mean Power (kW)")
    ax.plot(ma7.index, ma7.values, color=PALETTE[1],
            linewidth=2.0, label="7-Day Moving Average")
    ax.fill_between(series.index, series.values, alpha=0.08, color=PALETTE[0])

    ax.set_title("⚡ Energy Consumption Overview", fontsize=15, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Active Power (kW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    ax.legend()
    ax.grid(True)

    return _save(fig, "consumption_overview.png")


# ──────────────────────────────────────────────
# 2. SEASONAL PATTERNS HEATMAP
# ──────────────────────────────────────────────

def plot_seasonal_patterns(df: pd.DataFrame,
                           col: str = TARGET) -> Path:
    """Hour (y-axis) × Day-of-week (x-axis) mean power heatmap."""
    print("[VIZ] Plotting seasonal patterns heatmap …")
    df2 = df.copy()
    df2["hour"]      = df2.index.hour
    df2["dayofweek"] = df2.index.dayofweek
    pivot = df2.pivot_table(values=col, index="hour", columns="dayofweek", aggfunc="mean")
    pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor=BG_COLOR,
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Mean Power (kW)", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("🕒 Seasonal Load Patterns — Hour × Day of Week", fontsize=14, pad=12)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")
    ax.invert_yaxis()

    return _save(fig, "seasonal_patterns.png")


# ──────────────────────────────────────────────
# 3. FEATURE IMPORTANCE
# ──────────────────────────────────────────────

def plot_feature_importance(importance_dict: dict, top_n: int = 15) -> Path:
    """Horizontal bar chart of XGBoost feature importances."""
    print("[VIZ] Plotting feature importance …")
    df_imp = (
        pd.Series(importance_dict, name="Importance")
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    df_imp.columns = ["Feature", "Importance"]
    df_imp = df_imp.sort_values("Importance")   # ascending for left-to-right bars

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df_imp["Feature"], df_imp["Importance"],
                   color=PALETTE[0], edgecolor=BG_COLOR, height=0.6)
    # Gradient colour
    for i, bar in enumerate(bars):
        bar.set_alpha(0.5 + 0.5 * i / len(bars))

    ax.set_title("🔍 XGBoost — Top Feature Importances", fontsize=14, pad=12)
    ax.set_xlabel("Importance Score (gain)")
    ax.grid(axis="x", alpha=0.3)

    return _save(fig, "feature_importance.png")


# ──────────────────────────────────────────────
# 4. MODEL COMPARISON
# ──────────────────────────────────────────────

def plot_model_comparison(metrics_df: pd.DataFrame) -> Path:
    """Grouped bar chart comparing MAE, RMSE, MAPE across all models."""
    print("[VIZ] Plotting model comparison …")
    cols   = ["MAE", "RMSE", "MAPE"]
    models = metrics_df.index.tolist()
    x      = np.arange(len(cols))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (model, color) in enumerate(zip(models, PALETTE)):
        vals = [metrics_df.loc[model, c] for c in cols]
        bars = ax.bar(x + i * width, vals, width, label=model,
                      color=color, alpha=0.85, edgecolor=BG_COLOR)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("🏆 Model Performance Comparison", fontsize=14, pad=12)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(cols, fontsize=12)
    ax.set_ylabel("Score")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    return _save(fig, "model_comparison.png")


# ──────────────────────────────────────────────
# 5. LSTM ACTUAL vs FORECAST
# ──────────────────────────────────────────────

def plot_lstm_forecast(y_true, y_pred, index,
                       std_dev: float = None) -> Path:
    """Line chart: actual (blue) vs LSTM forecast (red) + optional shaded CI."""
    print("[VIZ] Plotting LSTM forecast …")
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(index, y_true, color=PALETTE[0], linewidth=1.5,
            label="Actual", zorder=3)
    ax.plot(index, y_pred, color=PALETTE[1], linewidth=1.5,
            linestyle="--", label="LSTM Forecast", zorder=3)

    if std_dev is not None:
        ax.fill_between(index,
                        np.array(y_pred) - 1.96 * std_dev,
                        np.array(y_pred) + 1.96 * std_dev,
                        color=PALETTE[1], alpha=0.12,
                        label="95% Prediction Interval")

    ax.set_title("🤖 LSTM — Actual vs Forecast", fontsize=14, pad=12)
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Active Power (kW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    ax.legend()
    ax.grid(True)

    return _save(fig, "lstm_forecast.png")


# ──────────────────────────────────────────────
# 6. ANOMALY DETECTION
# ──────────────────────────────────────────────

def plot_anomaly_detection(df: pd.DataFrame, col: str = TARGET,
                           z_thresh: float = 3.0) -> Path:
    """Overlay Z-score-detected anomaly points on the power time-series."""
    print("[VIZ] Plotting anomaly detection …")
    series = df[col].dropna()
    z      = (series - series.mean()) / series.std()
    anomaly_mask = z.abs() > z_thresh
    anomalies = series[anomaly_mask]

    # Use a 30-day window for readability
    last30 = series.last("30D")
    a30    = anomalies[anomalies.index >= last30.index.min()]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(last30.index, last30.values, color=PALETTE[0],
            linewidth=1.0, alpha=0.8, label="Power (kW)", zorder=2)
    ax.scatter(a30.index, a30.values, color=PALETTE[1], s=60,
               zorder=5, label=f"Anomaly (|Z| > {z_thresh})", marker="o")
    ax.fill_between(last30.index, last30.values, alpha=0.06, color=PALETTE[0])

    ax.set_title("🚨 Anomaly Detection — Last 30 Days", fontsize=14, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Active Power (kW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    ax.legend()
    ax.grid(True)

    total_anomalies = anomaly_mask.sum()
    ax.annotate(f"Total anomalies detected: {total_anomalies}",
                xy=(0.01, 0.96), xycoords="axes fraction",
                fontsize=10, color=PALETTE[1])

    return _save(fig, "anomaly_detection.png")


# ──────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────

def generate_all_charts(df: pd.DataFrame, metrics_df: pd.DataFrame = None,
                        importance_dict: dict = None,
                        y_true=None, y_pred=None) -> list:
    """
    Generate all 6 charts and return list of output paths.
    Pass None for optional params to skip those charts.
    """
    paths = []
    paths.append(plot_consumption_overview(df))
    paths.append(plot_seasonal_patterns(df))
    paths.append(plot_anomaly_detection(df))

    if importance_dict:
        paths.append(plot_feature_importance(importance_dict))
    if metrics_df is not None and not metrics_df.empty:
        paths.append(plot_model_comparison(metrics_df))
    if y_true is not None and y_pred is not None:
        idx = df.index[-len(y_true):]
        paths.append(plot_lstm_forecast(y_true, y_pred, idx))

    print(f"\n[VIZ] {len(paths)} charts saved to {IMAGES_DIR}")
    return paths
