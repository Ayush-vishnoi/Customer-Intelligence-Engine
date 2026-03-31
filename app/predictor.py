"""
predictor.py
------------
Loads all three saved models and exposes a single predict() function
used by the Streamlit app.

Given a customer's RFM values, returns:
  - segment name  (e.g. "High-Value")
  - churn probability  (0–1)
  - predicted CLV in £
"""

import os
import sys
import numpy as np
import pandas as pd

# Allow imports from the project root when running the app/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.segmentation import CustomerSegmentation
from src.churn_model  import ChurnModel
from src.clv_model    import CLVModel

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class Predictor:
    """
    Lazy-loading inference wrapper. Models are loaded once on first use.
    """

    def __init__(self):
        self._seg   = None
        self._churn = None
        self._clv   = None

    # ── public ──────────────────────────────────

    def predict(
        self,
        recency: float,
        frequency: float,
        monetary: float,
    ) -> dict:
        """
        Predict segment, churn probability, and CLV for a single customer.

        Parameters
        ----------
        recency   : float  Days since last purchase.
        frequency : float  Number of distinct orders.
        monetary  : float  Total spend in £.

        Returns
        -------
        dict with keys: segment, churn_probability, predicted_clv_gbp
        """
        self._load_models()
        row = self._build_row(recency, frequency, monetary)

        # Segment
        seg_result = self._seg.assign_segments(row)
        segment    = seg_result["Segment"].iloc[0]

        # Churn
        churn_prob = float(self._churn.predict_proba(row)[0])

        # CLV
        predicted_clv = float(self._clv.predict(row, in_pounds=True)[0])

        return {
            "segment":           segment,
            "churn_probability": round(churn_prob, 4),
            "predicted_clv_gbp": round(predicted_clv, 2),
        }

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run predictions on a DataFrame with columns:
        [CustomerID, Recency, Frequency, Monetary]

        Returns the input DataFrame with three new columns appended.
        """
        self._load_models()
        rows = pd.concat(
            [self._build_row(r.Recency, r.Frequency, r.Monetary)
             for r in df.itertuples()],
            ignore_index=True,
        )

        seg_rows  = self._seg.assign_segments(rows)
        churn_prob = self._churn.predict_proba(rows)
        clv_pred   = self._clv.predict(rows, in_pounds=True)

        out = df.copy()
        out["Segment"]           = seg_rows["Segment"].values
        out["ChurnProbability"]  = churn_prob.round(4)
        out["PredictedCLV_GBP"]  = clv_pred.round(2)
        return out

    # ── private ─────────────────────────────────

    def _load_models(self):
        if self._seg is None:
            self._seg   = CustomerSegmentation.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
            self._churn = ChurnModel.load(os.path.join(MODELS_DIR, "churn_rf_model.pkl"))
            self._clv   = CLVModel.load(os.path.join(MODELS_DIR, "clv_rf_model.pkl"))

    def _build_row(self, recency, frequency, monetary) -> pd.DataFrame:
        """Build a one-row feature DataFrame from raw RFM inputs."""
        import numpy as np
        avg_basket   = monetary / max(frequency, 1)
        total_items  = frequency * 5          # rough proxy when not available
        return pd.DataFrame([{
            "CustomerID":    0,
            "Recency":       recency,
            "Frequency":     frequency,
            "Monetary":      monetary,
            "AvgBasket":     avg_basket,
            "TotalItems":    total_items,
            "LogMonetary":   np.log1p(monetary),
            "LogAvgBasket":  np.log1p(avg_basket),
            "LogTotalItems": np.log1p(total_items),
        }])
