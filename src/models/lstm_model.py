"""
lstm_model.py
=============
LSTM deep learning forecaster using TensorFlow/Keras.

Architecture:
  Input  → LSTM(128) → Dropout(0.2)
         → LSTM(64)  → Dropout(0.2)
         → Dense(32) → ReLU
         → Dense(1)  → Linear (regression)

Sliding window: uses last `look_back` timesteps to predict next 1 step.

Usage:
    model = LSTMForecaster(look_back=24)
    model.fit(series)
    preds = model.predict(series)
    model.save() / model.load()
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _create_sequences(data: np.ndarray, look_back: int):
    """
    Convert 1-D array to (X, y) sliding-window sequences.

    Args:
        data      : Normalized 1-D numpy array.
        look_back : Number of past timesteps to use as input.

    Returns:
        X shape: (n_samples, look_back, 1)
        y shape: (n_samples,)
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back: i])
        y.append(data[i])
    return np.array(X).reshape(-1, look_back, 1), np.array(y)


class LSTMForecaster:
    """
    Two-layer stacked LSTM for univariate time-series forecasting.
    """

    def __init__(self, look_back: int = 24, units: list = None,
                 dropout: float = 0.2, epochs: int = 30,
                 batch_size: int = 64, learning_rate: float = 1e-3):
        self.look_back     = look_back
        self.units         = units or [128, 64]
        self.dropout       = dropout
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.model_        = None
        self.history_      = None
        self.scaler_       = None

    # ── Build Architecture ───────────────────────────────────
    def _build_model(self):
        """Build and compile the Keras model."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        model = Sequential([
            LSTM(self.units[0], input_shape=(self.look_back, 1),
                 return_sequences=True, name="lstm_1"),
            Dropout(self.dropout),
            LSTM(self.units[1], return_sequences=False, name="lstm_2"),
            Dropout(self.dropout),
            Dense(32, activation="relu", name="dense_1"),
            Dense(1, name="output"),
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="huber",
            metrics=["mae"],
        )
        return model

    # ── Fit ──────────────────────────────────────────────────
    def fit(self, series: pd.Series,
            val_split: float = 0.1) -> "LSTMForecaster":
        """
        Train the LSTM on a (normalized) univariate time series.

        For best results pass MinMax-scaled series in [0,1].

        Args:
            series    : pd.Series or np.ndarray of values.
            val_split : Fraction of data used for validation.
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        import tensorflow as tf

        tf.random.set_seed(42)
        np.random.seed(42)

        data = np.array(series, dtype=float)
        X, y = _create_sequences(data, self.look_back)

        # Temporal split (no shuffling)
        split = int(len(X) * (1 - val_split))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        print(f"\n[LSTM] Building model …")
        self.model_ = self._build_model()
        self.model_.summary(print_fn=lambda x: print("  " + x))

        print(f"\n[LSTM] Training: {len(X_tr):,} sequences   Val: {len(X_val):,}")

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", patience=4,
                              factor=0.5, min_lr=1e-6, verbose=1),
        ]
        self.history_ = self.model_.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        final_val_loss = min(self.history_.history["val_loss"])
        print(f"[LSTM] Best val_loss: {final_val_loss:.6f} ✓")
        return self

    # ── Predict ──────────────────────────────────────────────
    def predict(self, series: pd.Series) -> np.ndarray:
        """
        Generate one-step-ahead predictions for the entire series.

        Args:
            series : Same scale as training data.

        Returns:
            np.ndarray of predicted values (length = len(series) - look_back).
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        data       = np.array(series, dtype=float)
        X, _       = _create_sequences(data, self.look_back)
        predictions = self.model_.predict(X, verbose=0).flatten()
        return predictions

    def forecast_multi_step(self, seed_sequence: np.ndarray,
                            steps: int = 24) -> np.ndarray:
        """
        Recursive multi-step forecast using the last `look_back` values.

        Args:
            seed_sequence : np.ndarray of shape (look_back,) — recent context.
            steps         : Number of future steps to forecast.

        Returns:
            np.ndarray of shape (steps,).
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        window = list(seed_sequence[-self.look_back:])
        preds  = []
        for _ in range(steps):
            x   = np.array(window[-self.look_back:]).reshape(1, self.look_back, 1)
            out = self.model_.predict(x, verbose=0)[0, 0]
            preds.append(float(out))
            window.append(out)
        return np.array(preds)

    # ── Save / Load ──────────────────────────────────────────
    def save(self, name: str = "lstm_model.h5") -> Path:
        path = MODEL_DIR / name
        self.model_.save(str(path))
        print(f"[LSTM] Model saved → {path}")
        return path

    def load(self, name: str = "lstm_model.h5") -> "LSTMForecaster":
        from tensorflow.keras.models import load_model
        path = MODEL_DIR / name
        self.model_ = load_model(str(path))
        print(f"[LSTM] Model loaded ← {path}")
        return self

    def get_training_history(self) -> dict:
        """Return training loss/mae history as dict."""
        if self.history_ is None:
            return {}
        return {
            "loss"    : self.history_.history.get("loss", []),
            "val_loss": self.history_.history.get("val_loss", []),
            "mae"     : self.history_.history.get("mae", []),
            "val_mae" : self.history_.history.get("val_mae", []),
        }
