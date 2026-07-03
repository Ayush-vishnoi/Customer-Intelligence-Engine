"""
segmentation.py
---------------
K-Means customer segmentation using RFM features.

Workflow
--------
1. Scale features with StandardScaler
2. Select best k via silhouette score (k=2..8)
3. Fit final K-Means model
4. Assign human-readable segment labels based on cluster medians
5. Persist model to disk with joblib
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


SEGMENT_LABELS = {
    1: "High-Value",
    2: "Loyal",
    3: "At-Risk",
    4: "Low-Value",
}


class CustomerSegmentation:
    """
    K-Means customer segmentation wrapper.

    Parameters
    ----------
    k_range : tuple
        Range of k values to search (default: 2–8).
    random_state : int
        Reproducibility seed.
    """

    def __init__(self, k_range: tuple = (2, 9), random_state: int = 42):
        self.k_range = k_range
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None  # KMeans
        self.best_k = None  # int
        self.feature_cols = []

    # ── public methods ──────────────────────────────

    def fit(self, rfm: pd.DataFrame, feature_cols=None) -> "CustomerSegmentation":
        """
        Fit the segmentation model.

        Parameters
        ----------
        rfm : pd.DataFrame
            Feature table (output of features.build_rfm).
        feature_cols : list[str], optional
            Columns to cluster on. Defaults to ['Recency','Frequency','LogMonetary'].

        Returns
        -------
        self
        """
        if feature_cols is None:
            feature_cols = ["Recency", "Frequency", "LogMonetary"]
        self.feature_cols = feature_cols

        X = self.scaler.fit_transform(rfm[feature_cols])
        self.best_k, self.elbow_data = self._select_k(X)
        print(f"[segmentation] Best k = {self.best_k}")

        self.model = KMeans(
            n_clusters=self.best_k,
            random_state=self.random_state,
            n_init=10,
        )
        self.model.fit(X)
        return self

    def predict(self, rfm: pd.DataFrame) -> pd.Series:
        """Return cluster labels for the given RFM table."""
        self._check_fitted()
        X = self.scaler.transform(rfm[self.feature_cols])
        return pd.Series(self.model.predict(X), index=rfm.index, name="Cluster")

    def assign_segments(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'Cluster' and 'Segment' columns to rfm.

        Segment names are assigned by ranking clusters on median Monetary
        (highest = High-Value, next = Loyal, etc.).
        """
        self._check_fitted()
        rfm = rfm.copy()
        rfm["Cluster"] = self.predict(rfm)

        # Rank clusters by median monetary value
        medians = rfm.groupby("Cluster")["Monetary"].median().sort_values(ascending=False)
        label_map = {
            cluster: SEGMENT_LABELS.get(rank, f"Segment {rank}")
            for rank, cluster in enumerate(medians.index, start=1)
        }
        rfm["Segment"] = rfm["Cluster"].map(label_map)
        return rfm

    def cluster_summary(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Return per-cluster median RFM statistics."""
        self._check_fitted()
        cols = ["Cluster", "Segment", "Recency", "Frequency", "Monetary"]
        present = [c for c in cols if c in rfm.columns]
        return (
            rfm[present]
            .groupby(["Cluster", "Segment"] if "Segment" in present else ["Cluster"])
            .median()
            .round(2)
            .reset_index()
        )

    def save(self, path: str = "models/kmeans_model.pkl") -> None:
        """Persist scaler + model to disk."""
        self._check_fitted()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"scaler": self.scaler, "model": self.model,
                     "best_k": self.best_k, "feature_cols": self.feature_cols}, path)
        print(f"[segmentation] Model saved → {path}")

    @classmethod
    def load(cls, path: str = "models/kmeans_model.pkl") -> "CustomerSegmentation":
        """Load a previously saved model."""
        data = joblib.load(path)
        obj = cls()
        obj.scaler = data["scaler"]
        obj.model = data["model"]
        obj.best_k = data["best_k"]
        obj.feature_cols = data["feature_cols"]
        print(f"[segmentation] Model loaded ← {path}")
        return obj

    # ── private helpers ─────────────────────────────

    def _select_k(self, X: np.ndarray) -> tuple[int, dict]:
        inertias, silhouettes = [], []
        ks = list(range(*self.k_range))
        for k in ks:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X, labels))
        best_k = ks[int(np.argmax(silhouettes))]
        return best_k, {"k": ks, "inertia": inertias, "silhouette": silhouettes}

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
