"""
feature_engineer.py
===================
Creates rich ML-ready features from the cleaned energy DataFrame.

Feature Groups:
  1. Calendar features  — hour, day, month, season, is_weekend, is_holiday
  2. Lag features       — 1h, 2h, 6h, 12h, 24h, 48h, 168h (1 week)
  3. Rolling statistics — mean, std, min, max over 3h, 6h, 24h windows
  4. Interaction terms  — temperature × hour, occupancy × power
  5. Cyclical encoding  — sin/cos for hour and month (avoids ordinal bias)
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

TARGET = "Global_active_power"


# ──────────────────────────────────────────────
# 1. CALENDAR FEATURES
# ──────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index

    df["hour"]      = idx.hour
    df["dayofweek"] = idx.dayofweek        # 0=Mon, 6=Sun
    df["day"]       = idx.day
    df["month"]     = idx.month
    df["quarter"]   = idx.quarter
    df["year"]      = idx.year
    df["dayofyear"] = idx.day_of_year

    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_weekday"] = 1 - df["is_weekend"]

    # Season: 0=Winter, 1=Spring, 2=Summer, 3=Autumn
    df["season"] = pd.cut(
        idx.month,
        bins=[0, 3, 6, 9, 12],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)

    # Peak-hour flag (common utility billing windows)
    df["is_peak_hour"] = (
        ((idx.hour >= 7) & (idx.hour <= 10)) |
        ((idx.hour >= 18) & (idx.hour <= 22))
    ).astype(int)

    print(f"  Calendar features added: {['hour','dayofweek','month','season','is_weekend','is_peak_hour']}")
    return df


# ──────────────────────────────────────────────
# 2. CYCLICAL ENCODING
# ──────────────────────────────────────────────

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour and month as (sin, cos) pairs to preserve cyclical nature."""
    df = df.copy()
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    print("  Cyclical encodings added (hour, month, day-of-week)")
    return df


# ──────────────────────────────────────────────
# 3. LAG FEATURES
# ──────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, col: str = TARGET,
                     lags: list = None) -> pd.DataFrame:
    """
    Add lag (shifted) versions of the target column.
    Default lags cover short-term (1h) to weekly (168h) patterns.
    """
    df = df.copy()
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 168]

    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    print(f"  Lag features added: lags={lags}")
    return df


# ──────────────────────────────────────────────
# 4. ROLLING STATISTICS
# ──────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, col: str = TARGET,
                         windows: list = None) -> pd.DataFrame:
    """Add rolling mean, std, min, and max over specified window sizes."""
    df = df.copy()
    if windows is None:
        windows = [3, 6, 12, 24, 48]

    for w in windows:
        df[f"{col}_roll_mean_{w}"] = df[col].shift(1).rolling(w).mean()
        df[f"{col}_roll_std_{w}"]  = df[col].shift(1).rolling(w).std()
        df[f"{col}_roll_min_{w}"]  = df[col].shift(1).rolling(w).min()
        df[f"{col}_roll_max_{w}"]  = df[col].shift(1).rolling(w).max()

    print(f"  Rolling features added: windows={windows}")
    return df


# ──────────────────────────────────────────────
# 5. INTERACTION FEATURES
# ──────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-motivated interaction terms."""
    df = df.copy()

    if "temperature_c" in df.columns:
        df["temp_x_hour"] = df["temperature_c"] * df.get("hour", df.index.hour)
        # Heating/cooling degree hours
        df["hdd"] = np.maximum(0, 18 - df["temperature_c"])   # Heating Degree
        df["cdd"] = np.maximum(0, df["temperature_c"] - 18)   # Cooling Degree

    if "occupancy" in df.columns and TARGET in df.columns:
        df["occupancy_x_power"] = df["occupancy"] * df[TARGET]

    if "is_peak_hour" in df.columns and TARGET in df.columns:
        df["peak_power"] = df["is_peak_hour"] * df[TARGET]

    print("  Interaction features added (temp×hour, HDD/CDD, occupancy×power)")
    return df


# ──────────────────────────────────────────────
# 6. FULL PIPELINE
# ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, save: bool = True,
                      tag: str = "features") -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df   : Cleaned DataFrame (output of preprocessor).
        save : Whether to save to data/processed/.
        tag  : Filename tag for saved file.

    Returns:
        Feature-rich DataFrame (NaN rows from lags are dropped).
    """
    print("\n[FEATURES] Engineering features …")
    df = add_calendar_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_interaction_features(df)

    before = len(df)
    df.dropna(inplace=True)
    print(f"  Dropped {before - len(df):,} rows with NaN after lag/rolling")
    print(f"  Final shape: {df.shape}")

    if save:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        out = PROC_DIR / f"{tag}.csv"
        df.to_csv(out)
        print(f"  Saved → {out}")

    print("[FEATURES] Done ✓\n")
    return df


def get_feature_columns(df: pd.DataFrame) -> tuple:
    """
    Return (feature_cols, target_col) split.
    Excludes the raw sub-metering columns and the target itself.

    Returns:
        (X_cols: list, y_col: str)
    """
    exclude = {TARGET, "datetime"}
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols, TARGET


def train_test_split_temporal(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Temporally-aware train/test split (no shuffling — respects time order).

    Returns:
        (train_df, test_df)
    """
    n = len(df)
    split = int(n * (1 - test_ratio))
    train = df.iloc[:split]
    test  = df.iloc[split:]
    print(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    return train, test
