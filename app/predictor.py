"""
predictor.py — Model inference wrapper for Flask
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.segmentation import CustomerSegmentation
from src.churn_model  import ChurnModel
from src.clv_model    import CLVModel


class Predictor:
    def __init__(self, models_dir: str = "../models"):
        self.models_dir = models_dir
        self._seg   = None
        self._churn = None
        self._clv   = None

    def _load(self):
        if self._seg is None:
            self._seg   = CustomerSegmentation.load(os.path.join(self.models_dir, "kmeans_model.pkl"))
            self._churn = ChurnModel.load(os.path.join(self.models_dir, "churn_rf_model.pkl"))
            self._clv   = CLVModel.load(os.path.join(self.models_dir, "clv_rf_model.pkl"))

    def _build_row(self, recency, frequency, monetary) -> pd.DataFrame:
        avg_basket    = monetary / max(frequency, 1)
        total_items   = frequency * 5
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

    def predict(self, recency: float, frequency: float, monetary: float) -> dict:
        self._load()
        row = self._build_row(recency, frequency, monetary)

        seg_row   = self._seg.assign_segments(row)
        segment   = seg_row["Segment"].iloc[0]
        churn_prob = float(self._churn.predict_proba(row)[0])
        clv        = float(self._clv.predict(row, in_pounds=True)[0])

        if churn_prob >= 0.7:
            risk = "High"
        elif churn_prob >= 0.4:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "segment":           segment,
            "churn_probability": round(churn_prob, 4),
            "churn_risk":        risk,
            "predicted_clv_gbp": round(max(clv, 0), 2),
            "inputs": {
                "recency":   recency,
                "frequency": frequency,
                "monetary":  monetary,
            },
        }

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self._load()
        rows = pd.concat(
            [self._build_row(r.Recency, r.Frequency, r.Monetary)
             for r in df.itertuples()],
            ignore_index=True,
        )
        seg_rows   = self._seg.assign_segments(rows)
        churn_prob = self._churn.predict_proba(rows)
        clv_pred   = self._clv.predict(rows, in_pounds=True)

        out = df.copy()
        out["Segment"]          = seg_rows["Segment"].values
        out["ChurnProbability"] = churn_prob.round(4)
        out["PredictedCLV_GBP"] = np.maximum(clv_pred, 0).round(2)
        out["ChurnRisk"]        = pd.cut(
            out["ChurnProbability"],
            bins=[-0.01, 0.4, 0.7, 1.01],
            labels=["Low", "Medium", "High"],
        )
        return out
