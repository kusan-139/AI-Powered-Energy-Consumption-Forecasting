# Methodology — AI-Powered Energy Consumption Forecasting

## 1. Problem Statement

Energy providers need accurate short-term (1–168 hour) consumption forecasts to:
- Optimise grid dispatch and reduce peak load
- Minimise renewable energy wastage
- Enable demand-response pricing
- Detect faulty meters and consumption anomalies

This project builds an end-to-end ML pipeline to forecast household active power consumption.

---

## 2. Dataset

**Primary:** UCI Machine Learning Repository — Individual Household Electric Power Consumption  
- 2M+ minute-level readings from one household in Sceaux, France (2006–2010)
- Public domain, no sign-in required

**Augmentation:** Synthetic weather (temperature, humidity) and occupancy features generated using `src/simulator.py` to simulate real smart-grid context.

---

## 3. Preprocessing Pipeline

```
Raw Data → Missing Value Imputation → Outlier Handling → Hourly Resampling
```

| Step | Method | Rationale |
|------|--------|-----------|
| Missing value imputation | Forward-fill (max 60min) + linear interpolation | Short gaps = sensor glitches; interpolation preserves trend |
| Outlier handling | Z-score clipping (|Z| > 5) then re-interpolate | Preserves anomaly instances for detection module |
| Resampling | Hourly mean (power), Hourly sum (sub-metering) | Reduces noise, aligns with industry SCADA reporting period |
| Normalisation | MinMaxScaler [0,1] on LSTM target only | Required for stable LSTM training; other models are scale-invariant |

---

## 4. Feature Engineering

50+ features across 5 groups:

1. **Calendar** — hour, dayofweek, month, season, is_weekend, is_peak_hour  
2. **Cyclical** — sin/cos encoding of hour, month, day-of-week (eliminates ordinal bias)  
3. **Lag** — 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h (captures autocorrelation structure)  
4. **Rolling** — mean, std, min, max over 3h, 6h, 12h, 24h, 48h windows  
5. **Interaction** — temperature × hour, HDD, CDD, occupancy × power  

---

## 5. Models

### 5.1 ARIMA (Baseline)
- **Type:** SARIMA(1,1,1)(1,1,1,24) — statsmodels SARIMAX
- **Input:** Univariate hourly power series
- **Stationarity:** ADF test before fit; differencing order d=1 removes trend
- **Seasonal period s=24** captures daily patterns
- **Best for:** Short-term (1–24h) where statistical stationarity holds

### 5.2 XGBoost (Primary)
- **Type:** Gradient Boosting Regression (XGBRegressor)
- **Input:** Full 50+ feature matrix
- **Hyperparameters:** n_estimators=500, max_depth=6, lr=0.05, subsample=0.8
- **Validation:** TimeSeriesSplit (5-fold, no shuffle) + early stopping (30 rounds)
- **Interpretability:** Feature importance (gain) shows lag_24 dominates
- **Best for:** Daily operations — fast inference, interpretable

### 5.3 LSTM (Deep Learning)
- **Architecture:** LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)
- **Input:** 24-hour sliding window (look-back=24)
- **Loss:** Huber (robust to outliers vs MSE)
- **Callbacks:** EarlyStopping (patience=8), ReduceLROnPlateau (patience=4)
- **Best for:** Weekly/long-range forecasting, captures complex non-linear patterns

---

## 6. Evaluation

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | mean\|y−ŷ\| | Average absolute error in kW |
| RMSE | √mean(y−ŷ)² | Penalises large errors more |
| MAPE | 100×mean\|y−ŷ\|/y | Percentage error (scale-free) |
| SMAPE | Symmetric MAPE | Robust when y≈0 |
| R² | 1−SS_res/SS_tot | Fraction of variance explained |

**Split:** Temporal 80/20 (last 20% of time-ordered data = test set — no data leakage).

---

## 7. Anomaly Detection

**Method:** Z-score thresholding on the power series  
**Threshold:** |Z| > 3.0 (covers 99.73% of normal distribution)  
**Categories:**  
- |Z| > 3.0 → Low anomaly  
- |Z| > 4.0 → Medium  
- |Z| > 5.0 → High  
- |Z| > 6.0 → Critical  

**Business value:** Detects meter faults, unplanned high-load events, and grid disturbances.

---

## 8. Production Deployment Architecture

```
Data Source (Smart Meter / UCI CSV)
        ↓
  src/data_loader.py    ← ingest & validate
        ↓
  src/preprocessor.py   ← clean & resample
        ↓
  src/feature_engineer.py ← create features
        ↓
  ┌──────────────────────────────┐
  │  Model Selection             │
  │  ├── arima_model.py          │
  │  ├── xgboost_model.py        │
  │  └── lstm_model.py           │
  └──────────────────────────────┘
        ↓
  src/evaluator.py      ← metrics & comparison
        ↓
  src/visualizer.py     ← generate charts
        ↓
  src/reporter.py       ← PDF report
        ↓
  dashboard/app.py      ← Flask REST API + UI
```
