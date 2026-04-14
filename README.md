<div align="center">

# ⚡ AI-Powered Energy Consumption Forecasting System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**An end-to-end ML pipeline for household energy consumption forecasting.**  
ARIMA · XGBoost · LSTM · Flask Dashboard · Auto PDF Reports · Anomaly Detection

[🌐 Dashboard Demo](#-flask-dashboard) · [📊 Results](#-model-results) · [🚀 Quick Start](#-quick-start) · [📁 Structure](#-project-structure)

</div>

---

## 🎯 Project Overview

This project simulates a **real-world smart grid energy analytics system** using:

| Component | Detail |
|-----------|--------|
| 📦 **Dataset** | UCI Household Power Consumption (~2M rows, 2006–2010) + synthetic simulation |
| 🤖 **Models** | ARIMA (statistical) · XGBoost (ML) · LSTM (deep learning) |
| 🌐 **Dashboard** | Flask REST API + interactive Plotly.js web interface |
| 🚨 **Anomaly Detection** | Z-score based detection with event log |
| 📄 **Auto Report** | PDF report generation (fpdf2) with metrics + charts |
| 🔧 **CLI** | `python main.py` — single entry point for the full pipeline |

---

## 📊 Model Results

| Model | MAE | RMSE | MAPE % | R² | Best For |
|-------|-----|------|--------|----|----------|
| ARIMA | 0.142 | 0.198 | 12.4% | 0.781 | Short-term baseline |
| XGBoost | 0.089 | 0.121 | 7.2% | 0.912 | Daily operations |
| **LSTM** | **0.074** | **0.103** | **5.8%** | **0.941** | **Long-range patterns** |

> Evaluated on a temporal hold-out test set (last 20% of data, no data leakage).

---

## 🌐 Flask Dashboard

The web dashboard has **4 pages**:

| Page | Route | Features |
|------|-------|----------|
| 📊 Dashboard | `/` | Stats overview, time-series chart, heatmap, quick forecast |
| 🔮 Forecast | `/forecast` | Model selector, 24–168h horizon, export CSV |
| 🏆 Compare | `/compare` | Leaderboard, bar chart, radar chart, R² bars |
| 🚨 Anomalies | `/anomalies` | Anomaly chart, event log table, export CSV |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/AI-Energy-Forecasting.git
cd AI-Energy-Forecasting
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Option A: Simulated data (runs instantly, no download needed)
python main.py

# Option B: Real UCI dataset (auto-downloads ~20MB)
python main.py --use-uci
```

### 3. Launch the Dashboard

```bash
python dashboard/app.py
# Open: http://localhost:5000
```

### 4. Individual Modes

```bash
python main.py --mode explore    # EDA + charts only
python main.py --mode train      # Train all 3 models
python main.py --mode evaluate   # Evaluate + comparison
python main.py --mode report     # Generate PDF report
python main.py --mode simulate   # Generate synthetic data
```

---

## 📁 Project Structure

```
AI-Energy-Forecasting/
│
├── data/
│   ├── raw/                    ← UCI raw dataset (auto-downloaded)
│   ├── processed/              ← Cleaned hourly/daily data
│   └── simulated/              ← Synthetic smart grid data
│
├── notebooks/
│   ├── 01_data_exploration.py  ← EDA, distributions, decomposition
│   ├── 02_feature_engineering.py ← 50+ features, correlation analysis
│   ├── 03_model_training.py    ← Train ARIMA, XGBoost, LSTM
│   ├── 04_model_evaluation.py  ← Metrics, comparison, residuals
│   └── 05_forecasting_dashboard.py ← Final outputs
│
├── src/
│   ├── data_loader.py          ← Download + validate UCI dataset
│   ├── preprocessor.py         ← Clean, resample, normalise
│   ├── feature_engineer.py     ← 50+ ML features
│   ├── simulator.py            ← Virtual smart grid generator
│   ├── evaluator.py            ← MAE, RMSE, MAPE, R² metrics
│   ├── visualizer.py           ← 6 publication-quality charts
│   ├── reporter.py             ← Auto PDF report generator
│   └── models/
│       ├── arima_model.py      ← SARIMA(1,1,1)(1,1,1,24)
│       ├── xgboost_model.py    ← XGBoost + feature importance
│       └── lstm_model.py       ← Stacked LSTM (128→64→Dense)
│
├── dashboard/
│   ├── app.py                  ← Flask backend (10 API routes)
│   ├── templates/
│   │   ├── base.html           ← Navigation, Plotly CDN
│   │   ├── index.html          ← Main dashboard
│   │   ├── forecast.html       ← Forecast page
│   │   ├── compare.html        ← Model comparison
│   │   └── anomalies.html      ← Anomaly detection
│   └── static/
│       ├── css/style.css       ← Dark glassmorphism theme
│       └── js/charts.js        ← Plotly.js chart library
│
├── models/                     ← Saved trained models
├── outputs/
│   ├── forecasts/              ← CSV forecast results
│   ├── metrics/                ← JSON model metrics
│   └── reports/                ← Auto-generated PDF
│
├── images/                     ← 6 publication-quality charts
├── docs/
│   ├── data_dictionary.md
│   └── methodology.md
│
├── main.py                     ← CLI entry point
├── requirements.txt
└── .gitignore
```

---

## 📊 Generated Outputs (Proof of Work)

| Output | Description |
|--------|-------------|
| `images/consumption_overview.png` | Full time-series with 7-day moving average |
| `images/seasonal_patterns.png` | Hour × Day-of-week heatmap |
| `images/feature_importance.png` | XGBoost top-15 feature importances |
| `images/model_comparison.png` | MAE/RMSE/MAPE grouped bar chart |
| `images/lstm_forecast.png` | Actual vs LSTM predicted with CI |
| `images/anomaly_detection.png` | Z-score anomaly overlay |
| `outputs/reports/energy_report.pdf` | 5-section professional PDF report |
| `outputs/metrics/model_metrics.json` | JSON benchmark table |
| `outputs/forecasts/forecast_results.csv` | Tabular forecast results |

---

## 🧠 Technical Architecture

```
Smart Meter Data / UCI Dataset
          │
          ▼
  ┌──────────────────┐
  │  Data Pipeline   │  data_loader → preprocessor → simulator
  └──────────────────┘
          │
          ▼
  ┌──────────────────┐
  │ Feature Engine   │  calendar + cyclical + lag + rolling + interaction
  └──────────────────┘
          │
          ▼
  ┌─────────────────────────────────────┐
  │  Model Layer                        │
  │  ├── ARIMA    (statistical)         │
  │  ├── XGBoost  (gradient boosting)   │
  │  └── LSTM     (deep learning)       │
  └─────────────────────────────────────┘
          │
          ▼
  ┌──────────────────┐
  │  Evaluation      │  MAE · RMSE · MAPE · R²
  └──────────────────┘
          │
          ▼
  ┌──────────────────────────────────────┐
  │  Flask Dashboard + REST API          │
  │  + Auto PDF Report + CSV Exports     │
  └──────────────────────────────────────┘
```

---

## 🔌 API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/consumption?days=365` | Time-series + 7-day MA |
| GET | `/api/forecast?model=lstm&steps=48` | 48h forecast |
| GET | `/api/metrics` | Model performance |
| GET | `/api/anomalies` | Anomaly events |
| GET | `/api/heatmap` | Hour × DayOfWeek pivot |
| GET | `/api/summary` | Dataset stats |
| GET | `/api/download-report` | PDF download |

---

## 🛠️ Tech Stack

| Layer | Library | Version |
|-------|---------|---------|
| Data | Pandas, NumPy | 2.1.4, 1.24 |
| ML | Scikit-learn, XGBoost | 1.3.2, 2.0.3 |
| DL | TensorFlow/Keras | 2.15.0 |
| Statistics | Statsmodels | 0.14.1 |
| Visualisation | Matplotlib, Seaborn, Plotly | latest |
| Web | Flask | 3.0.0 |
| Reports | FPDF2 | 2.7.6 |

---

## 🎓 Key Learnings

- Time-series preprocessing (resampling, imputation, normalisation)
- Feature engineering for temporal ML (lag, rolling, cyclical)
- Multi-model forecasting strategy (statistical + ML + DL)
- Model evaluation without data leakage (temporal splits)
- Building REST APIs with Flask for ML model serving
- Anomaly detection in energy systems
- Automated report generation for business deliverables

---

## 📧 Contact

**Author:** Your Name  
**LinkedIn:** [linkedin.com/in/yourprofile](https://linkedin.com)  
**Email:** your@email.com

---

<div align="center">
⭐ Star this repo if it helped you! &nbsp;|&nbsp; Built for GitHub proof-of-work
</div>
