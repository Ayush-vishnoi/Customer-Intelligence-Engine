"""
clv_model.py
------------
Customer Lifetime Value (CLV) regression.

Target   : log1p(Monetary) — predicts expected total spend (log scale)
Models   : Linear Regression (baseline) | Random Forest Regressor (best)
Metric   : RMSE on log scale, R² score

Post-processing: inverse log transform predictions back to £ values.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class CLVModel:
    """
    Train and evaluate CLV regression models.

    Parameters
    ----------
    test_size : float
        Held-out fraction (default: 0.2).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.best_model = None
        self.best_model_name: str = ""
        self.feature_cols: list[str] = []
        self.target_col: str = "LogMonetary"
        self.results: dict = {}
        self._X_test = None
        self._y_test = None

    # ── public methods ──────────────────────────────

    def fit(self, rfm: pd.DataFrame, feature_cols: list[str] | None = None,
            target_col: str = "LogMonetary") -> "CLVModel":
        """
        Train all regressors and select the best by R².

        Parameters
        ----------
        rfm : pd.DataFrame
            Feature table with both feature columns and the target.
        feature_cols : list[str], optional
            Defaults to ['Frequency','Recency','LogAvgBasket','LogTotalItems'].
        target_col : str
            Log-transformed spend column (default: 'LogMonetary').
        """
        if feature_cols is None:
            feature_cols = ["Frequency", "Recency", "LogAvgBasket", "LogTotalItems"]
        self.feature_cols = feature_cols
        self.target_col = target_col

        X = rfm[feature_cols]
        y = rfm[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
        )
        self._X_test = X_test
        self._y_test = y_test

        candidates = self._build_candidates()
        best_r2 = -np.inf

        for name, model in candidates.items():
            print(f"[clv] Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)

            self.results[name] = {
                "RMSE (log)": rmse,
                "MAE (log)":  mae,
                "R²":         r2,
                "model":      model,
            }
            print(f"[clv]   {name}: RMSE={rmse:.4f}, R²={r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                self.best_model = model
                self.best_model_name = name

        print(f"[clv] Best model: {self.best_model_name} (R²={best_r2:.4f})")
        return self

    def predict(self, X: pd.DataFrame, in_pounds: bool = True) -> np.ndarray:
        """
        Predict CLV for new customers.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain feature_cols.
        in_pounds : bool
            If True, inverse-transform predictions back to £ values.

        Returns
        -------
        np.ndarray of predicted CLV values.
        """
        self._check_fitted()
        log_pred = self.best_model.predict(X[self.feature_cols])
        return np.expm1(log_pred) if in_pounds else log_pred

    def evaluation_report(self) -> pd.DataFrame:
        """Tidy comparison of all trained regressors."""
        rows = {
            name: {k: v for k, v in metrics.items() if k != "model"}
            for name, metrics in self.results.items()
        }
        return pd.DataFrame(rows).T.round(4)

    def feature_importances(self) -> pd.Series | None:
        """Feature importances for tree-based models; None otherwise."""
        self._check_fitted()
        if hasattr(self.best_model, "feature_importances_"):
            return pd.Series(
                self.best_model.feature_importances_,
                index=self.feature_cols,
            ).sort_values(ascending=False)
        return None

    def residuals(self) -> pd.DataFrame:
        """Return a DataFrame of actual vs predicted on the test set."""
        self._check_fitted()
        y_pred_log = self.best_model.predict(self._X_test)
        return pd.DataFrame({
            "actual_log":    self._y_test.values,
            "predicted_log": y_pred_log,
            "actual_gbp":    np.expm1(self._y_test.values),
            "predicted_gbp": np.expm1(y_pred_log),
            "residual":      self._y_test.values - y_pred_log,
        })

    def save(self, path: str = "models/clv_rf_model.pkl") -> None:
        """Persist the best model."""
        self._check_fitted()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model":        self.best_model,
            "model_name":   self.best_model_name,
            "feature_cols": self.feature_cols,
            "target_col":   self.target_col,
        }, path)
        print(f"[clv] Model saved → {path}")

    @classmethod
    def load(cls, path: str = "models/clv_rf_model.pkl") -> "CLVModel":
        """Load a previously saved CLV model."""
        data = joblib.load(path)
        obj = cls()
        obj.best_model      = data["model"]
        obj.best_model_name = data["model_name"]
        obj.feature_cols    = data["feature_cols"]
        obj.target_col      = data["target_col"]
        print(f"[clv] Model loaded ← {path}  ({obj.best_model_name})")
        return obj

    # ── private helpers ─────────────────────────────

    def _build_candidates(self) -> dict:
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=200, max_depth=10,
                random_state=self.random_state, n_jobs=-1,
            ),
        }

    def _check_fitted(self):
        if self.best_model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
