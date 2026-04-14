"""
evaluator.py
============
Model evaluation utilities.

Metrics computed:
  - MAE   : Mean Absolute Error
  - RMSE  : Root Mean Squared Error
  - MAPE  : Mean Absolute Percentage Error
  - R²    : Coefficient of Determination
  - SMAPE : Symmetric MAPE (robust to near-zero values)

Also provides:
  - compare_models() → comparison table DataFrame
  - save_metrics()   → persist to outputs/metrics/
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR    = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "outputs" / "metrics"


# ──────────────────────────────────────────────
# METRIC FUNCTIONS
# ──────────────────────────────────────────────

def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """MAPE — clips denominator to avoid division by zero."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-8)) * 100)


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def adj_r2(y_true, y_pred, n_features: int = 1) -> float:
    """Adjusted R-squared."""
    r2_val = r2_score(y_true, y_pred)
    n = len(y_true)
    if n <= n_features + 1:
        return float(r2_val)
    return float(1 - (1 - r2_val) * (n - 1) / (n - n_features - 1))


def evaluate_model(y_true, y_pred, model_name: str = "Model", n_features: int = 1) -> dict:
    """
    Compute all metrics for a single model.

    Args:
        y_true     : Ground-truth values (array-like).
        y_pred     : Predicted values (array-like).
        model_name : Name tag for the results dict.

    Returns:
        Dictionary with keys: model, MAE, RMSE, MAPE, SMAPE, R2
    """
    results = {
        "model" : model_name,
        "MAE"   : round(mae(y_true, y_pred), 4),
        "RMSE"  : round(rmse(y_true, y_pred), 4),
        "MAPE"  : round(mape(y_true, y_pred), 2),
        "SMAPE" : round(smape(y_true, y_pred), 2),
        "R2"    : round(r2(y_true, y_pred), 4),
        "Adj_R2": round(adj_r2(y_true, y_pred, n_features), 4),
    }

    print(f"\n  ┌─ {model_name} Results ─────────────────")
    for k, v in results.items():
        if k != "model":
            unit = "%" if k in ("MAPE", "SMAPE") else ""
            print(f"  │  {k:<8}: {v}{unit}")
    print("  └─────────────────────────────────────")

    return results


def compare_models(results_list: list) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a list of evaluate_model() dicts.

    Returns:
        pd.DataFrame sorted by RMSE ascending.
    """
    df = pd.DataFrame(results_list).set_index("model")
    df = df.sort_values("RMSE")
    print("\n[COMPARISON] Model Leaderboard:")
    print(df.to_string())
    return df


def save_metrics(results_list: list, filename: str = "model_metrics.json") -> Path:
    """Persist metrics to outputs/metrics/ as JSON."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / filename
    with open(out, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"\n[INFO] Metrics saved → {out}")
    return out


def load_metrics(filename: str = "model_metrics.json") -> list:
    """Load previously saved metrics."""
    path = METRICS_DIR / filename
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def residual_analysis(y_true, y_pred, model_name: str) -> dict:
    """
    Compute residual statistics for diagnostic reporting.

    Returns:
        Dict with mean, std, min, max of residuals.
    """
    residuals = np.array(y_true) - np.array(y_pred)
    return {
        "model"          : model_name,
        "residual_mean"  : round(float(residuals.mean()), 6),
        "residual_std"   : round(float(residuals.std()), 6),
        "residual_min"   : round(float(residuals.min()), 6),
        "residual_max"   : round(float(residuals.max()), 6),
    }
