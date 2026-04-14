"""
main.py
=======
CLI entry point — runs the complete AI Energy Forecasting pipeline.

Usage:
    python main.py                        # Full pipeline with simulated data
    python main.py --mode explore         # Data exploration + charts only
    python main.py --mode train           # Train all 3 models
    python main.py --mode evaluate        # Evaluate + compare models
    python main.py --mode forecast        # Generate 24h forecast
    python main.py --mode report          # Generate PDF report
    python main.py --mode full            # Everything end-to-end
    python main.py --use-uci             # Use real UCI dataset (auto-downloads)
    python main.py --freq D               # Daily granularity (default: H = hourly)
"""

import argparse
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ── Ensure src/ is on path ───────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Banner ───────────────────────────────────────────────────

BANNER = """
+============================================================+
|   AI-POWERED ENERGY CONSUMPTION FORECASTING SYSTEM        |
|      ARIMA  |  XGBoost  |  LSTM  |  Flask Dashboard       |
+============================================================+
"""


def print_banner():
    print(BANNER)


# ── Stage Functions ───────────────────────────────────────────

def stage_data(args):
    """Load / generate data and run preprocessing."""
    print("\n━━━ [1/6] DATA INGESTION ━━━")
    if args.use_uci:
        from src.data_loader import download_dataset, load_raw_data, validate_data, get_summary_stats
        filepath = download_dataset()
        df_raw   = load_raw_data(filepath)
        validate_data(df_raw)
        stats    = get_summary_stats(df_raw)

        from src.preprocessor import preprocess_uci
        df = preprocess_uci(df_raw, freq=args.freq)
    else:
        print("[INFO] Using simulated data (pass --use-uci for real UCI dataset)")
        from src.simulator import generate_synthetic_data
        from src.preprocessor import preprocess_simulated
        from src.data_loader import get_summary_stats
        df = generate_synthetic_data(freq=args.freq)
        df = preprocess_simulated(df)
        stats = {
            "source"       : "Simulated Smart Grid",
            "total_records": len(df),
            "date_from"    : str(df.index.min().date()),
            "date_to"      : str(df.index.max().date()),
            "avg_power_kw" : round(df["Global_active_power"].mean(), 4),
            "max_power_kw" : round(df["Global_active_power"].max(), 4),
        }

    print(f"  ✓ Data ready: {df.shape}")
    return df, stats


def stage_explore(df):
    """EDA + initial charts."""
    print("\n━━━ [2/6] EXPLORATION & VISUALISATION ━━━")
    from src.visualizer import plot_consumption_overview, plot_seasonal_patterns, plot_anomaly_detection
    plot_consumption_overview(df)
    plot_seasonal_patterns(df)
    plot_anomaly_detection(df)
    print("  ✓ Charts saved to images/")


def stage_features(df):
    """Feature engineering."""
    print("\n━━━ [3/6] FEATURE ENGINEERING ━━━")
    from src.feature_engineer import engineer_features, get_feature_columns, train_test_split_temporal
    df_feat = engineer_features(df)
    feat_cols, target = get_feature_columns(df_feat)
    train_df, test_df = train_test_split_temporal(df_feat)
    print(f"  ✓ {len(feat_cols)} features | Train={len(train_df):,} Test={len(test_df):,}")
    return df_feat, train_df, test_df, feat_cols, target


def stage_train(df, train_df, test_df, feat_cols, target, args):
    """Train all 3 models."""
    print("\n━━━ [4/6] MODEL TRAINING ━━━")

    results = []

    # ─── ARIMA ───────────────────────────────────────────────
    print("\n  ▶ ARIMA (SARIMA baseline) …")
    try:
        from src.models.arima_model import ARIMAForecaster
        arima_series = df["Global_active_power"].dropna()
        # Use last 2000 points only (ARIMA is slow on full data)
        arima_series = arima_series.iloc[-2000:]
        split_idx    = int(len(arima_series) * 0.8)
        arima_train  = arima_series.iloc[:split_idx]
        arima_test   = arima_series.iloc[split_idx:]

        arima = ARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
        arima.fit(arima_train)
        arima_preds = arima.predict(steps=len(arima_test))
        arima.save()

        from src.evaluator import evaluate_model
        arima_result = evaluate_model(arima_test.values, arima_preds, "ARIMA")
        results.append(arima_result)
    except Exception as e:
        print(f"  [WARN] ARIMA skipped: {e}")
        results.append({"model": "ARIMA", "MAE": 999, "RMSE": 999, "MAPE": 999, "R2": 0})

    # ─── XGBoost ─────────────────────────────────────────────
    print("\n  ▶ XGBoost …")
    from src.models.xgboost_model import XGBoostForecaster
    X_train = train_df[feat_cols]
    y_train = train_df[target]
    X_test  = test_df[feat_cols]
    y_test  = test_df[target]

    n_val = int(len(X_train) * 0.1)
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

    xgb = XGBoostForecaster()
    xgb.fit(X_tr, y_tr, X_val, y_val)
    xgb_preds    = xgb.predict(X_test)
    importance   = xgb.get_feature_importance()
    xgb.save()

    from src.evaluator import evaluate_model
    xgb_result = evaluate_model(y_test.values, xgb_preds, "XGBoost")
    results.append(xgb_result)

    # Feature importance chart
    from src.visualizer import plot_feature_importance
    plot_feature_importance(importance)

    # ─── LSTM ────────────────────────────────────────────────
    print("\n  ▶ LSTM …")
    from src.models.lstm_model import LSTMForecaster
    from sklearn.preprocessing import MinMaxScaler
    scaler_lstm = MinMaxScaler()
    series_raw  = df["Global_active_power"].dropna()
    series_norm = scaler_lstm.fit_transform(series_raw.values.reshape(-1, 1)).flatten()

    lstm = LSTMForecaster(look_back=24, epochs=args.lstm_epochs)
    lstm.fit(series_norm)
    lstm_preds_norm = lstm.predict(series_norm)
    lstm_preds = scaler_lstm.inverse_transform(
        lstm_preds_norm.reshape(-1, 1)).flatten()
    y_true_lstm = series_raw.values[24:]

    lstm.save()
    lstm_result = evaluate_model(y_true_lstm, lstm_preds, "LSTM")
    results.append(lstm_result)

    return results, xgb_preds, y_test, lstm_preds, y_true_lstm, importance, series_raw


def stage_evaluate(results, xgb_preds, y_test, lstm_preds, y_true_lstm, df):
    """Evaluate models + generate comparison chart."""
    print("\n━━━ [5/6] EVALUATION & COMPARISON ━━━")
    from src.evaluator import compare_models, save_metrics
    from src.visualizer import plot_model_comparison, plot_lstm_forecast

    metrics_df = compare_models(results)
    save_metrics(results)

    plot_model_comparison(metrics_df)
    plot_lstm_forecast(y_true_lstm[-200:], lstm_preds[-200:],
                       df.index[-200:])

    return metrics_df


def stage_report(stats, results):
    """Generate PDF report."""
    print("\n━━━ [6/6] REPORT GENERATION ━━━")
    from src.reporter import generate_report
    generate_report(stats=stats, metrics_list=results)


# ── Argument Parser ───────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AI-Powered Energy Consumption Forecasting Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "explore", "train", "evaluate", "forecast", "report", "simulate"],
        help=(
            "full     → Run entire pipeline (default)\n"
            "explore  → EDA + charts only\n"
            "train    → Train all models\n"
            "evaluate → Evaluate + compare models\n"
            "forecast → Generate 24h forecast\n"
            "report   → Generate PDF report\n"
            "simulate → Generate & save synthetic data only"
        ),
    )
    p.add_argument("--use-uci",  action="store_true",
                   help="Download and use real UCI dataset (requires internet)")
    p.add_argument("--freq",     type=str, default="H", choices=["H", "D"],
                   help="Data frequency: H=hourly (default), D=daily")
    p.add_argument("--lstm-epochs", type=int, default=20,
                   help="LSTM training epochs (default: 20)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────

def main():
    print_banner()
    args = parse_args()
    print(f"  Mode      : {args.mode.upper()}")
    print(f"  Data      : {'UCI Real Dataset' if args.use_uci else 'Simulated'}")
    print(f"  Frequency : {args.freq}\n")

    # ── Simulate only ─────────────────────────────────────────
    if args.mode == "simulate":
        from src.simulator import generate_synthetic_data
        generate_synthetic_data(freq=args.freq)
        print("\n✅ Simulation complete.")
        return

    # ── Load data ─────────────────────────────────────────────
    df, stats = stage_data(args)

    # ── Explore ───────────────────────────────────────────────
    if args.mode in ("full", "explore"):
        stage_explore(df)
    if args.mode == "explore":
        print("\n✅ Exploration complete.")
        return

    # ── Features ──────────────────────────────────────────────
    df_feat, train_df, test_df, feat_cols, target = stage_features(df)

    # ── Train ─────────────────────────────────────────────────
    if args.mode in ("full", "train", "evaluate"):
        results, xgb_preds, y_test, lstm_preds, y_true_lstm, importance, series_raw = \
            stage_train(df, train_df, test_df, feat_cols, target, args)
    if args.mode == "train":
        print("\n✅ Training complete.")
        return

    # ── Evaluate ──────────────────────────────────────────────
    if args.mode in ("full", "evaluate"):
        metrics_df = stage_evaluate(results, xgb_preds, y_test,
                                    lstm_preds, y_true_lstm, df)
    if args.mode == "evaluate":
        print("\n✅ Evaluation complete.")
        return

    # ── Report ────────────────────────────────────────────────
    if args.mode in ("full", "report"):
        stage_report(stats, results)

    print("\n" + "═" * 60)
    print("✅ PIPELINE COMPLETE")
    print("  📊 Charts   → images/")
    print("  🤖 Models   → models/")
    print("  📄 Report   → outputs/reports/energy_report.pdf")
    print("  📈 Metrics  → outputs/metrics/model_metrics.json")
    print("  🌐 Dashboard→ python dashboard/app.py")
    print("═" * 60)


if __name__ == "__main__":
    main()
