"""
dashboard/app.py
================
Flask web application for the AI Energy Consumption Forecasting dashboard.

Routes:
    GET  /                      → Main dashboard (overview + live stats)
    GET  /forecast              → Forecasting page (model selector)
    GET  /compare               → Model comparison page
    GET  /anomalies             → Anomaly detection page
    GET  /api/consumption       → JSON: time-series data
    GET  /api/forecast?model=X  → JSON: forecast for chosen model
    GET  /api/metrics           → JSON: model performance metrics
    GET  /api/anomalies         → JSON: detected anomaly events
    GET  /api/summary           → JSON: dataset summary stats
    GET  /api/heatmap           → JSON: hour × day-of-week pivot
    GET  /api/download-report   → Stream PDF report download
    POST /api/run-pipeline      → Trigger background pipeline run
"""

import sys
import json
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, send_file, Response

# ── path setup ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.simulator import generate_synthetic_data, load_simulated_data

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────
_df_cache: pd.DataFrame = None
_metrics_cache: list    = []
_pipeline_status: dict  = {"running": False, "stage": "idle", "progress": 0}


# ─────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────

def get_data() -> pd.DataFrame:
    """Load (or generate) the energy dataset, cached in memory."""
    global _df_cache
    if _df_cache is not None:
        return _df_cache

    # Try multiple locations in priority order
    candidates = [
        ROOT / "data" / "processed" / "processed_H.csv",
        ROOT / "data" / "processed" / "processed_simulated.csv",
        ROOT / "data" / "simulated" / "synthetic_energy_h.csv",
        ROOT / "data" / "simulated" / "synthetic_energy_H.csv",
    ]

    for path in candidates:
        if path.exists():
            _df_cache = pd.read_csv(path, index_col="datetime", parse_dates=True)
            print(f"[DASHBOARD] Data loaded from {path.name}: {_df_cache.shape}")
            return _df_cache

    print("[DASHBOARD] No data found --- generating synthetic data ...")
    _df_cache = generate_synthetic_data(freq="H", save=True)
    print(f"[DASHBOARD] Data loaded: {_df_cache.shape}")
    return _df_cache


def get_metrics() -> list:
    """Load model metrics from disk (if available)."""
    global _metrics_cache
    if _metrics_cache:
        return _metrics_cache

    path = ROOT / "outputs" / "metrics" / "model_metrics.json"
    if path.exists():
        with open(path) as f:
            _metrics_cache = json.load(f)
    else:
        # Return placeholder metrics for demo
        _metrics_cache = [
            {"model": "ARIMA",   "MAE": 0.142, "RMSE": 0.198, "MAPE": 12.4, "SMAPE": 11.8, "R2": 0.781, "Adj_R2": 0.771},
            {"model": "XGBoost", "MAE": 0.089, "RMSE": 0.121, "MAPE":  7.2, "SMAPE":  6.9, "R2": 0.912, "Adj_R2": 0.908},
            {"model": "LSTM",    "MAE": 0.074, "RMSE": 0.103, "MAPE":  5.8, "SMAPE":  5.5, "R2": 0.941, "Adj_R2": 0.935},
        ]
    return _metrics_cache


def detect_anomalies(series: pd.Series, z_thresh: float = 3.0) -> pd.DataFrame:
    """Z-score anomaly detection."""
    z = (series - series.mean()) / series.std()
    mask = z.abs() > z_thresh
    anomalies = series[mask].reset_index()
    anomalies.columns = ["datetime", "value"]
    anomalies["z_score"] = z[mask].values
    return anomalies


def simple_forecast(series: pd.Series, model: str = "xgboost",
                    steps: int = 48) -> dict:
    """
    Fast in-memory forecast for dashboard (no saved model required).
    Uses last 168h as baseline + seasonality pattern.
    """
    last_week = series.iloc[-168:].values if len(series) >= 168 else series.values
    last_day  = series.iloc[-24:].values  if len(series) >= 24  else series.values

    # Forecast = weighted mix of last-week same hour + last-day trend
    forecast = []
    for i in range(steps):
        week_val = last_week[i % len(last_week)]
        day_val  = last_day[i % len(last_day)]
        if model == "arima":
            val = 0.7 * week_val + 0.3 * day_val
        elif model == "xgboost":
            val = 0.5 * week_val + 0.5 * day_val
        else:  # lstm
            val = 0.4 * week_val + 0.6 * day_val
        noise = np.random.normal(0, 0.03 * abs(val))
        forecast.append(max(0, val + noise))

    start      = series.index[-1] + timedelta(hours=1)
    date_range = pd.date_range(start=start, periods=steps, freq="H")
    return {
        "timestamps": [t.strftime("%Y-%m-%d %H:%M") for t in date_range],
        "values"    : [round(float(v), 4) for v in forecast],
        "model"     : model,
        "steps"     : steps,
    }


# ─────────────────────────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    df    = get_data()
    gp    = df["Global_active_power"].dropna()
    stats = {
        "total_records" : f"{len(df):,}",
        "date_from"     : str(df.index.min().date()),
        "date_to"       : str(df.index.max().date()),
        "avg_power"     : f"{gp.mean():.3f} kW",
        "max_power"     : f"{gp.max():.3f} kW",
        "total_energy"  : f"{(gp.sum() / 1000):.1f} MWh",
        "anomaly_count" : str(int(((gp - gp.mean()).abs() / gp.std() > 3).sum())),
    }
    return render_template("index.html", stats=stats)


@app.route("/forecast")
def forecast_page():
    return render_template("forecast.html")


@app.route("/compare")
def compare_page():
    metrics = get_metrics()
    # Sort metrics so the best performing model (lowest MAE) is ranked #1
    sorted_metrics = sorted(metrics, key=lambda x: x.get("MAE", float("inf")))
    return render_template("compare.html", metrics=sorted_metrics)


@app.route("/anomalies")
def anomalies_page():
    return render_template("anomalies.html")


# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/api/consumption")
def api_consumption():
    """Return daily mean power for the full time-series chart."""
    df     = get_data()
    daily  = df["Global_active_power"].resample("D").mean().dropna()

    # Optional: last N days filter
    days = request.args.get("days", default=365, type=int)
    daily = daily.iloc[-days:]

    ma7 = daily.rolling(7, center=True).mean()
    return jsonify({
        "timestamps" : [t.strftime("%Y-%m-%d") for t in daily.index],
        "values"     : [round(float(v), 4) if not np.isnan(v) else None for v in daily.values],
        "ma7"        : [round(float(v), 4) if not np.isnan(v) else None for v in ma7.values],
        "unit"       : "kW",
    })


@app.route("/api/forecast")
def api_forecast():
    """Return 48-hour forecast for selected model."""
    model = request.args.get("model", default="xgboost").lower()
    steps = request.args.get("steps", default=48, type=int)
    df    = get_data()
    series = df["Global_active_power"].dropna()

    # Include last 48 actual hours for chart context
    last_actual = series.iloc[-48:]
    result = simple_forecast(series, model=model, steps=steps)
    return jsonify({
        "actual_timestamps" : [t.strftime("%Y-%m-%d %H:%M") for t in last_actual.index],
        "actual_values"     : [round(float(v), 4) for v in last_actual.values],
        "forecast_timestamps": result["timestamps"],
        "forecast_values"   : result["values"],
        "model"             : model,
    })


@app.route("/api/metrics")
def api_metrics():
    """Return model performance metrics."""
    return jsonify(get_metrics())


@app.route("/api/anomalies")
def api_anomalies():
    """Return detected anomaly events."""
    df     = get_data()
    series = df["Global_active_power"].dropna()

    # Detect anomalies over the full dataset to match global counts
    anomalies_df = detect_anomalies(series)

    # Filter for display
    days = request.args.get("days", default=0, type=int)
    if days > 0:
        cutoff = series.index[-1] - pd.Timedelta(days=days)
        series = series[series.index >= cutoff]
        anomalies_df = anomalies_df[anomalies_df["datetime"] >= cutoff]

    # Sample full series for chart background (every 12h = ~3000 points)
    step = 6 if (days > 0 and days <= 730) else 12
    normal_sample = series.iloc[::step]

    return jsonify({
        "series_timestamps": [t.strftime("%Y-%m-%d %H:%M") for t in normal_sample.index],
        "series_values"    : [round(float(v), 4) for v in normal_sample.values],
        "anomaly_timestamps": anomalies_df["datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        "anomaly_values"   : anomalies_df["value"].round(4).tolist(),
        "anomaly_zscores"  : anomalies_df["z_score"].round(2).tolist(),
        "total_anomalies"  : len(anomalies_df),
        "days"             : days,
    })


@app.route("/api/summary")
def api_summary():
    """Return key dataset statistics."""
    df = get_data()
    gp = df["Global_active_power"].dropna()
    return jsonify({
        "total_records" : len(df),
        "date_from"     : str(df.index.min().date()),
        "date_to"       : str(df.index.max().date()),
        "avg_power_kw"  : round(float(gp.mean()), 4),
        "max_power_kw"  : round(float(gp.max()), 4),
        "min_power_kw"  : round(float(gp.min()), 4),
        "std_power_kw"  : round(float(gp.std()), 4),
        "total_kwh"     : round(float(gp.sum()), 2),
    })


@app.route("/api/heatmap")
def api_heatmap():
    """Return hour × day-of-week mean power pivot for heatmap."""
    df = get_data()
    df2 = df.copy()
    df2["hour"]      = df2.index.hour
    df2["dayofweek"] = df2.index.dayofweek
    pivot = df2.pivot_table(
        values="Global_active_power", index="hour",
        columns="dayofweek", aggfunc="mean"
    ).round(4)
    return jsonify({
        "hours"   : list(range(24)),
        "days"    : ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "matrix"  : pivot.values.tolist(),
    })


@app.route("/api/download-report")
def download_report():
    """Stream the auto-generated PDF report."""
    report_path = ROOT / "outputs" / "reports" / "energy_report.pdf"
    if not report_path.exists():
        # Auto-generate if missing
        try:
            from src.reporter import generate_report
            metrics = get_metrics()
            df = get_data()
            gp = df["Global_active_power"].dropna()
            stats = {
                "Total Records": len(df),
                "Date From"    : str(df.index.min().date()),
                "Date To"      : str(df.index.max().date()),
                "Avg Power"    : f"{gp.mean():.3f} kW",
                "Max Power"    : f"{gp.max():.3f} kW",
            }
            generate_report(stats=stats, metrics_list=metrics)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return send_file(
        str(report_path),
        as_attachment=True,
        download_name="energy_forecasting_report.pdf",
        mimetype="application/pdf",
    )


@app.route("/api/pipeline-status")
def pipeline_status():
    return jsonify(_pipeline_status)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  AI Energy Forecasting Dashboard")
    print("  URL : http://127.0.0.1:5000")
    print("  Press CTRL+C to stop")
    print("=" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
