"""
churn_model.py
--------------
Binary churn prediction: will a customer stop buying?

Models trained
--------------
- Logistic Regression  (baseline, interpretable)
- Random Forest        (handles non-linearity, feature importance)
- XGBoost              (optional, best performance — install xgboost to enable)

The best model (by F1 on the test set) is stored as self.best_model.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score,
)

try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    _XGB = False


class ChurnModel:
    """
    Train and evaluate multiple churn classifiers, then keep the best.

    Parameters
    ----------
    test_size : float
        Fraction of data held out for evaluation (default: 0.2).
    random_state : int
        Reproducibility seed.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.best_model = None
        self.best_model_name: str = ""
        self.feature_cols: list[str] = []
        self.results: dict = {}
        self._X_test = None
        self._y_test = None

    # ── public methods ──────────────────────────────

    def fit(self, rfm: pd.DataFrame, feature_cols: list[str] | None = None,
            target_col: str = "Churned") -> "ChurnModel":
        """
        Train all classifiers and select the best by F1 score.

        Parameters
        ----------
        rfm : pd.DataFrame
            Must contain feature columns and the target column.
        feature_cols : list[str], optional
            Defaults to ['Frequency','LogMonetary','LogAvgBasket','LogTotalItems'].
        target_col : str
            Binary churn label column (default: 'Churned').
        """
        if feature_cols is None:
            feature_cols = ["Frequency", "LogMonetary", "LogAvgBasket", "LogTotalItems"]
        self.feature_cols = feature_cols

        X = rfm[feature_cols]
        y = rfm[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y,
        )
        self._X_test = X_test
        self._y_test = y_test

        candidates = self._build_candidates()
        best_f1 = -1.0

        for name, model in candidates.items():
            print(f"[churn] Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            report = classification_report(y_test, y_pred, output_dict=True)
            f1     = report["1"]["f1-score"]
            auc    = roc_auc_score(y_test, y_prob)

            self.results[name] = {
                "Accuracy":  report["accuracy"],
                "Precision": report["1"]["precision"],
                "Recall":    report["1"]["recall"],
                "F1":        f1,
                "ROC-AUC":   auc,
                "model":     model,
            }

            if f1 > best_f1:
                best_f1 = f1
                self.best_model = model
                self.best_model_name = name

        print(f"[churn] Best model: {self.best_model_name} (F1={best_f1:.3f})")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions using the best model."""
        self._check_fitted()
        return self.best_model.predict(X[self.feature_cols])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return churn probability (class=1) for each customer."""
        self._check_fitted()
        return self.best_model.predict_proba(X[self.feature_cols])[:, 1]

    def evaluation_report(self) -> pd.DataFrame:
        """Return a tidy comparison DataFrame of all trained models."""
        rows = {
            name: {k: v for k, v in metrics.items() if k != "model"}
            for name, metrics in self.results.items()
        }
        return pd.DataFrame(rows).T.round(3)

    def feature_importances(self) -> pd.Series | None:
        """
        Return feature importances for tree-based best models.
        Returns None for Logistic Regression.
        """
        self._check_fitted()
        if hasattr(self.best_model, "feature_importances_"):
            return pd.Series(
                self.best_model.feature_importances_,
                index=self.feature_cols,
            ).sort_values(ascending=False)
        return None

    def confusion(self) -> np.ndarray:
        """Return confusion matrix on the held-out test set."""
        self._check_fitted()
        y_pred = self.best_model.predict(self._X_test)
        return confusion_matrix(self._y_test, y_pred)

    def save(self, path: str = "models/churn_rf_model.pkl") -> None:
        """Persist the best model and metadata."""
        self._check_fitted()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model":        self.best_model,
            "model_name":   self.best_model_name,
            "feature_cols": self.feature_cols,
        }, path)
        print(f"[churn] Model saved → {path}")

    @classmethod
    def load(cls, path: str = "models/churn_rf_model.pkl") -> "ChurnModel":
        """Load a previously saved churn model."""
        data = joblib.load(path)
        obj = cls()
        obj.best_model      = data["model"]
        obj.best_model_name = data["model_name"]
        obj.feature_cols    = data["feature_cols"]
        print(f"[churn] Model loaded ← {path}  ({obj.best_model_name})")
        return obj

    # ── private helpers ─────────────────────────────

    def _build_candidates(self) -> dict:
        candidates = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=8,
                random_state=self.random_state, n_jobs=-1),
        }
        if _XGB:
            candidates["XGBoost"] = XGBClassifier(
                use_label_encoder=False, eval_metric="logloss",
                random_state=self.random_state, n_jobs=-1)
        return candidates

    def _check_fitted(self):
        if self.best_model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
