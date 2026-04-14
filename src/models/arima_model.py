"""
arima_model.py
==============
ARIMA/SARIMA baseline forecaster using statsmodels SARIMAX.

Usage:
    model = ARIMAForecaster()
    model.fit(train_series)
    preds = model.predict(steps=24)
    model.save() / model.load()
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ARIMAForecaster:
    """
    SARIMA(p,d,q)(P,D,Q,s) wrapper with auto-stationarity check.

    Default order (1,1,1)(1,1,1,24) works well for hourly energy data.
    """

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24),
                 enforce_stationarity=False, enforce_invertibility=False):
        self.order             = order
        self.seasonal_order    = seasonal_order
        self.enforce_s         = enforce_stationarity
        self.enforce_i         = enforce_invertibility
        self.model_            = None
        self.result_           = None
        self.train_index_      = None

    # ── Stationarity Check ───────────────────────────────────
    def check_stationarity(self, series: pd.Series) -> bool:
        """Augmented Dickey-Fuller test. Returns True if stationary."""
        result = adfuller(series.dropna())
        pval   = result[1]
        print(f"  ADF p-value: {pval:.4f} → {'Stationary ✓' if pval < 0.05 else 'Non-stationary — differencing needed'}")
        return pval < 0.05

    # ── Fit ──────────────────────────────────────────────────
    def fit(self, series: pd.Series, verbose: bool = True) -> "ARIMAForecaster":
        """
        Fit SARIMA model to a univariate time series.

        Args:
            series : pd.Series with datetime index (hourly frequency).
        """
        print(f"\n[ARIMA] Fitting SARIMA{self.order}x{self.seasonal_order} …")
        self.check_stationarity(series)
        self.train_index_ = series.index

        self.model_ = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_s,
            enforce_invertibility=self.enforce_i,
        )
        self.result_ = self.model_.fit(disp=False, maxiter=200)

        if verbose:
            print(f"  AIC  : {self.result_.aic:.2f}")
            print(f"  BIC  : {self.result_.bic:.2f}")
            print("[ARIMA] Fit complete ✓")

        return self

    # ── Predict ──────────────────────────────────────────────
    def predict(self, steps: int = 24) -> np.ndarray:
        """
        Forecast `steps` periods ahead.

        Returns:
            np.ndarray of predicted values (original scale).
        """
        if self.result_ is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        forecast = self.result_.forecast(steps=steps)
        return np.array(forecast)

    def predict_in_sample(self) -> np.ndarray:
        """Return in-sample fitted values."""
        return np.array(self.result_.fittedvalues)

    # ── Save / Load ──────────────────────────────────────────
    def save(self, name: str = "arima_model.pkl") -> Path:
        path = MODEL_DIR / name
        joblib.dump({"result": self.result_,
                     "order": self.order,
                     "seasonal_order": self.seasonal_order}, path)
        print(f"[ARIMA] Model saved → {path}")
        return path

    def load(self, name: str = "arima_model.pkl") -> "ARIMAForecaster":
        path = MODEL_DIR / name
        data = joblib.load(path)
        self.result_        = data["result"]
        self.order          = data["order"]
        self.seasonal_order = data["seasonal_order"]
        print(f"[ARIMA] Model loaded ← {path}")
        return self

    # ── Summary ──────────────────────────────────────────────
    def summary(self) -> str:
        if self.result_:
            return str(self.result_.summary())
        return "Model not fitted."
