"""
xgboost_model.py
================
XGBoost gradient-boosting forecaster with feature importance.

Uses the full feature-engineered DataFrame (lag, rolling, calendar, etc.)
treats forecasting as a supervised regression problem.

Usage:
    model = XGBoostForecaster()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    importance = model.get_feature_importance()
    model.save() / model.load()
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class XGBoostForecaster:
    """
    XGBoost regression wrapper with cross-validation, feature
    importance extraction, and SHAP-ready design.
    """

    DEFAULT_PARAMS = {
        "n_estimators"    : 500,
        "max_depth"       : 6,
        "learning_rate"   : 0.05,
        "subsample"       : 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha"       : 0.1,
        "reg_lambda"      : 1.0,
        "random_state"    : 42,
        "n_jobs"          : -1,
        "tree_method"     : "hist",
    }

    def __init__(self, params: dict = None):
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model_         = XGBRegressor(**p)
        self.feature_names_ = None
        self.params_        = p

    # ── Fit ──────────────────────────────────────────────────
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            early_stopping: int = 30) -> "XGBoostForecaster":
        """
        Train XGBoost on feature-engineered training data.

        Args:
            X_train        : Feature matrix.
            y_train        : Target series (Global_active_power).
            X_val / y_val  : Optional validation set for early stopping.
            early_stopping : Rounds of no improvement before stopping.
        """
        print(f"\n[XGBOOST] Training on {X_train.shape[0]:,} samples, "
              f"{X_train.shape[1]} features …")
        self.feature_names_ = list(X_train.columns)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model_.set_params(early_stopping_rounds=early_stopping)

        self.model_.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

        train_score = self.model_.score(X_train, y_train)
        print(f"  Train R²          : {train_score:.4f}")
        if eval_set:
            val_score = self.model_.score(X_val, y_val)
            print(f"  Validation R²     : {val_score:.4f}")
            print(f"  Best iteration    : {self.model_.best_iteration}")

        print("[XGBOOST] Training complete ✓")
        return self

    # ── Cross-Validation ─────────────────────────────────────
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       cv: int = 5) -> dict:
        """Time-series-safe cross-validation (no shuffling)."""
        from sklearn.model_selection import TimeSeriesSplit
        tscv   = TimeSeriesSplit(n_splits=cv)
        scores = cross_val_score(
            self.model_, X, y,
            cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        rmse_scores = -scores
        result = {
            "cv_rmse_mean": round(rmse_scores.mean(), 4),
            "cv_rmse_std" : round(rmse_scores.std(),  4),
        }
        print(f"  [CV] RMSE: {result['cv_rmse_mean']:.4f} ± {result['cv_rmse_std']:.4f}")
        return result

    # ── Predict ──────────────────────────────────────────────
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    # ── Feature Importance ───────────────────────────────────
    def get_feature_importance(self, importance_type: str = "gain") -> dict:
        """
        Return feature importances as {feature_name: score} dict.

        Args:
            importance_type : 'gain', 'weight', or 'cover'.
        """
        scores = self.model_.get_booster().get_score(importance_type=importance_type)
        # Fill missing features with 0
        full = {f: scores.get(f, 0.0) for f in (self.feature_names_ or [])}
        return dict(sorted(full.items(), key=lambda x: x[1], reverse=True))

    # ── Save / Load ──────────────────────────────────────────
    def save(self, name: str = "xgboost_model.pkl") -> Path:
        path = MODEL_DIR / name
        joblib.dump({"model": self.model_,
                     "features": self.feature_names_,
                     "params": self.params_}, path)
        print(f"[XGBOOST] Model saved → {path}")
        return path

    def load(self, name: str = "xgboost_model.pkl") -> "XGBoostForecaster":
        path = MODEL_DIR / name
        data = joblib.load(path)
        self.model_         = data["model"]
        self.feature_names_ = data["features"]
        self.params_        = data["params"]
        print(f"[XGBOOST] Model loaded ← {path}")
        return self
