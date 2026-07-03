"""
features.py
-----------
Builds the RFM (Recency, Frequency, Monetary) feature table plus
additional derived features from the cleaned transaction DataFrame.

Features created
----------------
Recency      – days since customer's last purchase (lower = more recent)
Frequency    – number of distinct invoices
Monetary     – total spend (£)
AvgBasket    – mean spend per invoice
TotalItems   – total units purchased
LogMonetary  – log1p(Monetary)   — reduces right skew for ML
LogAvgBasket – log1p(AvgBasket)
LogTotalItems– log1p(TotalItems)
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def build_rfm(df: pd.DataFrame, snapshot_date: "pd.Timestamp | None" = None) -> pd.DataFrame:
    """
    Compute RFM + derived features for each customer.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction data (output of data_loader.load_and_clean).
    snapshot_date : pd.Timestamp, optional
        Reference date for recency calculation.
        Defaults to one day after the latest InvoiceDate in the data.

    Returns
    -------
    pd.DataFrame
        One row per customer with all engineered features.
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    print(f"[features] Snapshot date: {snapshot_date.date()}")

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency    = ("InvoiceDate",  lambda x: (snapshot_date - x.max()).days),
            Frequency  = ("InvoiceNo",    "nunique"),
            Monetary   = ("TotalPrice",   "sum"),
            AvgBasket  = ("TotalPrice",   "mean"),
            TotalItems = ("Quantity",     "sum"),
        )
        .reset_index()
    )

    # Log-transform skewed features
    for col in ["Monetary", "AvgBasket", "TotalItems"]:
        rfm[f"Log{col}"] = np.log1p(rfm[col])

    print(f"[features] RFM table shape: {rfm.shape}")
    return rfm


def add_churn_label(rfm: pd.DataFrame, threshold_days: int = 90) -> pd.DataFrame:
    """
    Add a binary 'Churned' column.

    A customer is considered churned if their Recency exceeds
    `threshold_days`.

    Parameters
    ----------
    rfm : pd.DataFrame
        Output of build_rfm.
    threshold_days : int
        Inactivity threshold in days (default: 90).

    Returns
    -------
    pd.DataFrame
        rfm with an added 'Churned' column (0 = active, 1 = churned).
    """
    rfm = rfm.copy()
    rfm["Churned"] = (rfm["Recency"] > threshold_days).astype(int)
    churn_rate = rfm["Churned"].mean() * 100
    print(f"[features] Churn label added — threshold={threshold_days}d, rate={churn_rate:.1f}%")
    return rfm


def get_feature_sets() -> dict:
    """
    Return named feature lists used across the models.

    Returns
    -------
    dict with keys: 'segmentation', 'churn', 'clv'
    """
    return {
        "segmentation": ["Recency", "Frequency", "LogMonetary"],
        "churn":        ["Frequency", "LogMonetary", "LogAvgBasket", "LogTotalItems"],
        "clv":          ["Frequency", "Recency", "LogAvgBasket", "LogTotalItems"],
    }
