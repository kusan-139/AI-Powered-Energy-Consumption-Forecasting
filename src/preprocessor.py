"""
preprocessor.py
===============
Cleans the raw UCI dataset and resamples it to hourly/daily granularity.

Steps:
  1. Forward-fill short gaps (< 1 hour)
  2. Linear interpolation for medium gaps
  3. Drop rows where > 50% columns are missing
  4. Resample to hourly mean / daily sum
  5. Normalize target column
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"


# ──────────────────────────────────────────────
# 1. CLEANING
# ──────────────────────────────────────────────

def drop_high_missing_rows(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Remove rows where more than `threshold` fraction of values are NaN."""
    mask = df.isnull().mean(axis=1) < threshold
    dropped = (~mask).sum()
    if dropped:
        print(f"  Dropped {dropped:,} rows with >{threshold*100:.0f}% missing values")
    return df[mask]


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: forward-fill up to 60 min, then linear interpolation."""
    before = df.isnull().sum().sum()
    df = df.ffill(limit=60)          # forward fill up to 60 minutes
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.bfill()                  # back-fill any remaining at edges
    after = df.isnull().sum().sum()
    print(f"  Missing values: {before:,} → {after:,}")
    return df


def remove_outliers(df: pd.DataFrame, col: str = "Global_active_power",
                    z_thresh: float = 5.0) -> pd.DataFrame:
    """
    Replace extreme outliers (beyond z_thresh standard deviations) with NaN,
    then re-interpolate — preserving anomaly-event integrity for detection module.
    """
    z = (df[col] - df[col].mean()) / df[col].std()
    before = (z.abs() > z_thresh).sum()
    df.loc[z.abs() > z_thresh, col] = np.nan
    df[col] = df[col].interpolate(method="linear")
    print(f"  Outliers clipped in '{col}': {before:,} values")
    return df


# ──────────────────────────────────────────────
# 2. RESAMPLING
# ──────────────────────────────────────────────

def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute-level UCI data to hourly:
      - power columns → mean
      - sub_metering  → sum (total energy in Wh)
    """
    agg = {col: "mean" for col in df.columns}
    for col in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        if col in df.columns:
            agg[col] = "sum"

    hourly = df.resample("H").agg(agg)
    print(f"  Resampled: {len(df):,} min-level rows → {len(hourly):,} hourly rows")
    return hourly


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data to daily totals/means."""
    daily = df.resample("D").agg({
        col: "mean" for col in df.columns
    })
    print(f"  Resampled: {len(df):,} hourly rows → {len(daily):,} daily rows")
    return daily


# ──────────────────────────────────────────────
# 3. NORMALISATION
# ──────────────────────────────────────────────

def scale_target(df: pd.DataFrame, col: str = "Global_active_power",
                 save_scaler: bool = True) -> tuple:
    """
    MinMax-scale the target column to [0, 1].

    Returns:
        (scaled_df, scaler) — pass scaler to inverse_transform at inference.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.copy()
    df[[col]] = scaler.fit_transform(df[[col]])

    if save_scaler:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, MODEL_DIR / "target_scaler.pkl")
        print(f"  Scaler saved → {MODEL_DIR}/target_scaler.pkl")

    return df, scaler


# ──────────────────────────────────────────────
# 4. FULL PIPELINE
# ──────────────────────────────────────────────

def preprocess_uci(df: pd.DataFrame, freq: str = "H",
                   save: bool = True) -> pd.DataFrame:
    """
    Full pre-processing pipeline for the raw UCI DataFrame.

    Args:
        df   : Raw DataFrame from data_loader.load_raw_data()
        freq : Target frequency — 'H' (hourly) or 'D' (daily)
        save : Whether to save processed data to disk

    Returns:
        Clean, resampled DataFrame ready for feature engineering.
    """
    print("\n[PREPROCESS] Starting pipeline …")

    df = drop_high_missing_rows(df)
    df = fill_missing(df)
    df = remove_outliers(df)

    if freq == "H":
        df = resample_hourly(df)
    elif freq == "D":
        df = resample_daily(df)

    df = fill_missing(df)   # second pass after resampling

    if save:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        out = PROC_DIR / f"processed_{freq}.csv"
        df.to_csv(out)
        print(f"  Saved → {out}")

    print("[PREPROCESS] Done ✓\n")
    return df


def preprocess_simulated(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Lightweight preprocessor for already-clean simulated data.
    Handles any NaNs and validates types.
    """
    print("\n[PREPROCESS] Processing simulated data …")
    df = df.copy()
    df = fill_missing(df)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = fill_missing(df)

    if save:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        out = PROC_DIR / "processed_simulated.csv"
        df.to_csv(out)
        print(f"  Saved → {out}")

    print("[PREPROCESS] Done ✓\n")
    return df


def load_processed(freq: str = "H", simulated: bool = False) -> pd.DataFrame:
    """Load previously saved processed data."""
    filename = "processed_simulated.csv" if simulated else f"processed_{freq}.csv"
    path = PROC_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}\nRun preprocess first.")
    df = pd.read_csv(path, index_col="datetime", parse_dates=True)
    print(f"[INFO] Loaded processed data -> {path}  shape={df.shape}")
    return df
