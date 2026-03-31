"""
main.py
-------
Run the complete E-Commerce Customer Intelligence pipeline.

Usage
-----
    python main.py

Steps
-----
1. Load & clean data
2. Build RFM features + churn label
3. Customer segmentation  → models/kmeans_model.pkl
4. Churn prediction       → models/churn_rf_model.pkl
5. CLV regression         → models/clv_rf_model.pkl
6. Generate all charts    → reports/figures/
7. Print business insights
"""

import os
import pandas as pd

from src.data_loader  import load_and_clean, get_summary
from src.features     import build_rfm, add_churn_label, get_feature_sets
from src.segmentation import CustomerSegmentation
from src.churn_model  import ChurnModel
from src.clv_model    import CLVModel
from src.visualize    import Visualizer

# ── Config ────────────────────────────────────────────────
DATA_PATH         = "data/Online_Retail.xlsx"
CHURN_THRESHOLD   = 90           # days of inactivity = churned
PROCESSED_DIR     = "data/processed"
MODELS_DIR        = "models"
FIGURES_DIR       = "reports/figures"
# ──────────────────────────────────────────────────────────


def run_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,    exist_ok=True)
    os.makedirs(FIGURES_DIR,   exist_ok=True)

    viz = Visualizer(save_dir=FIGURES_DIR)
    feat_sets = get_feature_sets()

    # ── 1. Load & clean ──────────────────────────
    print("\n" + "="*55)
    print("  STEP 1 — DATA LOADING & CLEANING")
    print("="*55)
    df = load_and_clean(DATA_PATH)
    summary = get_summary(df)
    print(f"\nDataset summary:")
    for k, v in summary.items():
        print(f"  {k:<18}: {v}")

    # ── 2. EDA charts ────────────────────────────
    print("\n" + "="*55)
    print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
    print("="*55)
    viz.eda_dashboard(df)

    # ── 3. Feature engineering ───────────────────
    print("\n" + "="*55)
    print("  STEP 3 — FEATURE ENGINEERING")
    print("="*55)
    rfm = build_rfm(df)
    rfm = add_churn_label(rfm, threshold_days=CHURN_THRESHOLD)
    rfm.to_csv(f"{PROCESSED_DIR}/rfm_features.csv", index=False)
    print(f"[main] RFM saved → {PROCESSED_DIR}/rfm_features.csv")

    # ── 4. Segmentation ──────────────────────────
    print("\n" + "="*55)
    print("  STEP 4 — CUSTOMER SEGMENTATION")
    print("="*55)
    seg = CustomerSegmentation()
    seg.fit(rfm, feature_cols=feat_sets["segmentation"])
    rfm = seg.assign_segments(rfm)

    viz.cluster_selection(seg.elbow_data)
    viz.customer_segments(rfm)
    viz.segment_revenue_share(rfm)

    rfm.to_csv(f"{PROCESSED_DIR}/customer_segments.csv", index=False)
    seg.save(f"{MODELS_DIR}/kmeans_model.pkl")

    print("\nCluster summary:")
    print(seg.cluster_summary(rfm).to_string(index=False))

    # ── 5. Churn prediction ──────────────────────
    print("\n" + "="*55)
    print("  STEP 5 — CHURN PREDICTION")
    print("="*55)
    churn = ChurnModel()
    churn.fit(rfm, feature_cols=feat_sets["churn"])
    churn.save(f"{MODELS_DIR}/churn_rf_model.pkl")

    eval_df = churn.evaluation_report()
    print("\nModel comparison:")
    print(eval_df.to_string())

    viz.churn_model_comparison(eval_df)

    fi = churn.feature_importances()
    if fi is not None:
        viz.churn_feature_importance(fi)

    viz.confusion_matrix_plot(churn.confusion())

    # ── 6. CLV regression ────────────────────────
    print("\n" + "="*55)
    print("  STEP 6 — CUSTOMER LIFETIME VALUE (CLV)")
    print("="*55)
    clv = CLVModel()
    clv.fit(rfm, feature_cols=feat_sets["clv"])
    clv.save(f"{MODELS_DIR}/clv_rf_model.pkl")

    print("\nRegression results:")
    print(clv.evaluation_report().to_string())

    viz.clv_actual_vs_predicted(clv.residuals())

    fi_clv = clv.feature_importances()
    if fi_clv is not None:
        viz.clv_feature_importance(fi_clv)

    # ── 7. Business insights ─────────────────────
    print("\n" + "="*55)
    print("  STEP 7 — KEY BUSINESS INSIGHTS")
    print("="*55)

    viz.pareto_curve(rfm)

    total_rev   = rfm["Monetary"].sum()
    rfm_sorted  = rfm.sort_values("Monetary", ascending=False).copy()
    rfm_sorted["CumRevPct"] = rfm_sorted["Monetary"].cumsum() / total_rev * 100
    rfm_sorted["CumCustPct"] = range(1, len(rfm_sorted) + 1)
    rfm_sorted["CumCustPct"] = rfm_sorted["CumCustPct"] / len(rfm_sorted) * 100
    pct_80 = rfm_sorted.loc[rfm_sorted["CumRevPct"] <= 80, "CumCustPct"].max()

    inactive = (rfm["Recency"] > CHURN_THRESHOLD).mean() * 100
    q75      = rfm["Frequency"].quantile(0.75)
    high_mult = rfm[rfm["Frequency"] >= q75]["Monetary"].mean()
    low_mult  = rfm[rfm["Frequency"] <  q75]["Monetary"].mean()

    print(f"\n  Pareto   : {pct_80:.1f}% of customers → 80% of revenue")
    print(f"  Churn    : {inactive:.1f}% inactive > {CHURN_THRESHOLD} days")
    print(f"  Loyalty  : High-frequency buyers spend {high_mult/low_mult:.1f}x more")

    seg_rev = rfm.groupby("Segment")["Monetary"].sum() / total_rev * 100
    print("\n  Segment revenue share:")
    for seg_name, pct in seg_rev.sort_values(ascending=False).items():
        print(f"    {seg_name:<15}: {pct:.1f}%")

    # ── Done ─────────────────────────────────────
    print("\n" + "="*55)
    print("  PIPELINE COMPLETE")
    print("="*55)
    print(f"  Charts  → {FIGURES_DIR}/")
    print(f"  Models  → {MODELS_DIR}/")
    print(f"  Data    → {PROCESSED_DIR}/")
    print()


if __name__ == "__main__":
    run_pipeline()
