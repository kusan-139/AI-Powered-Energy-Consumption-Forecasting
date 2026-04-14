"""
simulator.py
============
Virtual Smart Grid Simulator — generates synthetic energy consumption data
that mimics real household/grid patterns when the UCI dataset is unavailable.

Features of simulated data:
  - Realistic daily load curves (morning & evening peaks)
  - Weekly seasonality (weekday vs weekend)
  - Annual seasonality (summer AC load, winter heating)
  - Temperature correlation
  - Occupancy patterns
  - Random noise + stubbed anomaly events
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
SIM_DIR  = BASE_DIR / "data" / "simulated"


def _temperature_profile(index: pd.DatetimeIndex) -> np.ndarray:
    """Generate synthetic temperature (°C) with annual seasonality."""
    day_of_year = index.day_of_year
    # Sinusoidal annual cycle: peak in July (~day 195), trough in January
    base_temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    # Daily cycle: cooler at night, warmer in afternoon
    hour_effect = 3 * np.sin(2 * np.pi * (index.hour - 6) / 24)
    noise = np.random.normal(0, 1.5, len(index))
    return np.round(base_temp + hour_effect + noise, 2)


def _occupancy_profile(index: pd.DatetimeIndex) -> np.ndarray:
    """Generate binary-ish occupancy signal (0=empty, 1=occupied)."""
    hour = index.hour
    is_weekend = (index.dayofweek >= 5).astype(float)

    # Weekday: home 0–8 (sleep), away 9–17 (work), home 18–23
    weekday_occ = np.where(
        (hour < 8) | (hour >= 18), 1.0,
        np.where(hour < 9, 0.5, 0.1)
    )
    # Weekend: mostly home
    weekend_occ = np.where(hour < 7, 0.8, 0.95)

    occupancy = weekday_occ * (1 - is_weekend) + weekend_occ * is_weekend
    noise = np.random.uniform(-0.05, 0.05, len(index))
    return np.clip(occupancy + noise, 0, 1)


def _load_curve(index: pd.DatetimeIndex, temperature: np.ndarray,
                occupancy: np.ndarray) -> np.ndarray:
    """
    Simulate Global_active_power (kW) using load-curve model.
    Base load + temperature-driven HVAC + occupancy × appliances + noise.
    """
    hour      = index.hour
    is_weekend = (index.dayofweek >= 5).astype(float)
    month     = index.month

    # Base load (always-on devices)
    base = 0.25 + 0.05 * np.sin(2 * np.pi * hour / 24)

    # Morning peak (6–9 AM)
    morning = 0.8 * np.exp(-0.5 * ((hour - 7.5) / 1.2) ** 2)
    # Evening peak (18–22 PM)
    evening = 1.2 * np.exp(-0.5 * ((hour - 19.5) / 1.8) ** 2)

    # HVAC: heating when cold (<10°C), cooling when hot (>26°C)
    hvac = np.where(temperature < 10, 0.6 * (10 - temperature) / 10,
           np.where(temperature > 26, 0.8 * (temperature - 26) / 10, 0))

    # Appliance load correlated with occupancy
    appliance = 0.5 * occupancy + 0.2 * is_weekend

    # Combine
    total = base + morning + evening + hvac + appliance
    noise = np.random.normal(0, 0.08, len(index))
    return np.clip(total + noise, 0.05, None).round(4)


def inject_anomalies(series: pd.Series, n_spikes: int = 20,
                     n_drops: int = 10) -> pd.Series:
    """
    Inject synthetic anomaly events into the power series.
    - Spikes: sudden surge (equipment fault / EV charging)
    - Drops:  sudden drop (outage / sensor failure)
    Returns modified series (anomaly indices are preserved in metadata).
    """
    s = series.copy()
    idx = np.random.choice(len(s), n_spikes + n_drops, replace=False)

    spike_idx = idx[:n_spikes]
    drop_idx  = idx[n_spikes:]

    s.iloc[spike_idx] *= np.random.uniform(2.5, 4.0, n_spikes)
    s.iloc[drop_idx]  *= np.random.uniform(0.0, 0.2, n_drops)

    return s.round(4)


def generate_synthetic_data(
    start: str = "2020-01-01",
    end:   str = "2023-12-31",
    freq:  str = "H",
    add_anomalies: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate a synthetic smart-meter dataset covering [start, end].

    Args:
        start         : Start date string (YYYY-MM-DD).
        end           : End date string (YYYY-MM-DD).
        freq          : Frequency — 'H' (hourly) or 'D' (daily).
        add_anomalies : Whether to inject anomaly events.
        save          : Whether to save the result to data/simulated/.

    Returns:
        pd.DataFrame with datetime index and energy + contextual features.
    """
    np.random.seed(42)
    # Normalise legacy freq aliases (H->h, D->d) for pandas 2.x
    freq_map = {'H': 'h', 'D': 'd', 'T': 'min', 'M': 'ME'}
    freq = freq_map.get(freq, freq)
    index = pd.date_range(start=start, end=end, freq=freq)
    print(f"[INFO] Generating synthetic data: {len(index):,} records ({freq}) …")

    temperature = _temperature_profile(index)
    occupancy   = _occupancy_profile(index)
    power       = _load_curve(index, temperature, occupancy)

    if add_anomalies:
        power_series = pd.Series(power, index=index)
        power_series = inject_anomalies(power_series)
        power = power_series.values

    # Sub-metering breakdown (kitchen, laundry, HVAC) — proportional split
    sub1 = np.clip(power * np.random.uniform(0.10, 0.20, len(index)), 0, None)
    sub2 = np.clip(power * np.random.uniform(0.08, 0.18, len(index)), 0, None)
    sub3 = np.clip(power * np.random.uniform(0.15, 0.30, len(index)), 0, None)

    df = pd.DataFrame({
        "Global_active_power"   : power,
        "temperature_c"         : temperature,
        "humidity_pct"          : np.clip(
            60 + 10 * np.sin(2 * np.pi * index.hour / 24) +
            np.random.normal(0, 5, len(index)), 20, 100).round(1),
        "occupancy"             : occupancy.round(3),
        "Sub_metering_1"        : sub1.round(4),
        "Sub_metering_2"        : sub2.round(4),
        "Sub_metering_3"        : sub3.round(4),
        "is_weekend"            : (index.dayofweek >= 5).astype(int),
        "is_holiday"            : 0,   # placeholder — can be enriched
    }, index=index)

    df.index.name = "datetime"

    if save:
        SIM_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SIM_DIR / f"synthetic_energy_{freq}.csv"
        df.to_csv(out_path)
        print(f"[SUCCESS] Synthetic data saved -> {out_path}")

    return df


def load_simulated_data(freq: str = "H") -> pd.DataFrame:
    """Load previously generated synthetic data or generate fresh."""
    # Normalise freq alias for file lookup
    freq_map = {'H': 'h', 'D': 'd', 'T': 'min', 'M': 'ME'}
    freq_norm = freq_map.get(freq, freq)
    path = SIM_DIR / f"synthetic_energy_{freq_norm}.csv"
    if not path.exists():
        # Fallback: try original key
        path = SIM_DIR / f"synthetic_energy_{freq}.csv"
    if path.exists():
        print(f"[INFO] Loading cached simulated data -> {path}")
        df = pd.read_csv(path, index_col="datetime", parse_dates=True)
        return df
    return generate_synthetic_data(freq=freq, save=True)
