"""
data_loader.py
==============
Handles downloading and loading the UCI Household Power Consumption dataset.
Dataset: https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

Columns:
    - Global_active_power   : Household global minute-averaged active power (kW)
    - Global_reactive_power : Household global minute-averaged reactive power (kW)
    - Voltage               : Minute-averaged voltage (V)
    - Global_intensity      : Household global minute-averaged current intensity (A)
    - Sub_metering_1        : Kitchen energy (Wh)
    - Sub_metering_2        : Laundry room energy (Wh)
    - Sub_metering_3        : Climate control energy (Wh)
"""

import os
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00235/household_power_consumption.zip"
)
BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
SIM_DIR    = BASE_DIR / "data" / "simulated"
# ─────────────────────────────────────────────────────────────


def download_dataset(url: str = UCI_URL, save_dir: Path = RAW_DIR) -> Path:
    """
    Download and extract the UCI dataset if not already present.

    Args:
        url      : Remote URL to download from.
        save_dir : Local directory to save raw data.

    Returns:
        Path to the extracted .txt file.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    zip_path = save_dir / "household_power_consumption.zip"
    txt_path = save_dir / "household_power_consumption.txt"

    if txt_path.exists():
        print(f"[INFO] Dataset already present → {txt_path}")
        return txt_path

    print("[INFO] Downloading UCI Household Power Consumption dataset…")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        desc="  Downloading",
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

    print("[INFO] Extracting…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(save_dir)
    zip_path.unlink()

    print(f"[SUCCESS] Dataset saved → {txt_path}")
    return txt_path


def load_raw_data(filepath: Path = None) -> pd.DataFrame:
    """
    Load the raw UCI dataset into a datetime-indexed DataFrame.

    Args:
        filepath : Path to the .txt file (auto-detected if None).

    Returns:
        pd.DataFrame with datetime index and 7 numeric feature columns.
    """
    if filepath is None:
        filepath = RAW_DIR / "household_power_consumption.txt"

    if not filepath.exists():
        raise FileNotFoundError(
            f"[ERROR] Dataset not found at {filepath}.\n"
            "Run download_dataset() first or use the simulator."
        )

    print(f"[INFO] Loading raw data from {filepath} …")
    df = pd.read_csv(
        filepath,
        sep=";",
        parse_dates={"datetime": ["Date", "Time"]},
        dayfirst=True,
        na_values=["?"],
        infer_datetime_format=True,
        low_memory=False,
    )
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")

    print(f"  Shape       : {df.shape}")
    print(f"  Date range  : {df.index.min()} → {df.index.max()}")
    missing_pct = df.isnull().mean().mean() * 100
    print(f"  Missing vals: {missing_pct:.2f}%")

    return df


def validate_data(df: pd.DataFrame) -> list:
    """
    Validate loaded DataFrame for expected structure and quality.

    Returns:
        List of issue strings (empty if all OK).
    """
    expected = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]
    issues = []

    for col in expected:
        if col not in df.columns:
            issues.append(f"Missing column: {col}")

    if df.index.duplicated().any():
        issues.append(f"Duplicate timestamps: {df.index.duplicated().sum():,}")

    if df.isnull().mean().mean() > 0.05:
        issues.append("Missing values exceed 5% threshold")

    if issues:
        print("[WARNING] Validation issues:")
        for i in issues:
            print(f"  ✗ {i}")
    else:
        print("[SUCCESS] Data validation passed ✓")

    return issues


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return a dictionary of key summary statistics for the dashboard."""
    gp = df["Global_active_power"].dropna()
    return {
        "total_records"     : len(df),
        "date_from"         : str(df.index.min().date()),
        "date_to"           : str(df.index.max().date()),
        "avg_power_kw"      : round(gp.mean(), 4),
        "max_power_kw"      : round(gp.max(), 4),
        "min_power_kw"      : round(gp.min(), 4),
        "missing_pct"       : round(df.isnull().mean().mean() * 100, 2),
        "total_kwh"         : round((gp / 60).sum(), 2),   # minute data → kWh
    }
