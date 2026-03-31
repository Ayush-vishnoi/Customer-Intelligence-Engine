"""
visualize.py
------------
All plotting functions for the E-Commerce Customer Intelligence project.

Each method saves a .png to the reports/figures/ directory AND
returns the matplotlib Figure so notebooks can display it inline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

PALETTE   = ["#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#4CC9F0"]
SAVE_DIR  = "reports/figures"

sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({"figure.dpi": 120, "axes.titleweight": "bold", "axes.titlesize": 13})


class Visualizer:
    """Centralised plotting class. All charts saved to reports/figures/."""

    def __init__(self, save_dir: str = SAVE_DIR):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    # ── EDA ────────────────────────────────────────

    def eda_dashboard(self, df: pd.DataFrame) -> plt.Figure:
        """4-panel EDA overview: revenue trend, top products, top countries, spend dist."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("E-Commerce EDA Dashboard", fontsize=15, y=1.01)

        # Monthly revenue
        monthly = (df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalPrice"]
                     .sum().reset_index())
        monthly["InvoiceDate"] = monthly["InvoiceDate"].astype(str)
        axes[0, 0].plot(monthly["InvoiceDate"], monthly["TotalPrice"] / 1e3,
                        marker="o", color=PALETTE[0], linewidth=2)
        axes[0, 0].set_title("Monthly Revenue (£k)")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"£{x:.0f}k"))

        # Top 10 products
        top_p = df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(10)
        axes[0, 1].barh(top_p.index[::-1], top_p.values[::-1] / 1e3, color=PALETTE[1])
        axes[0, 1].set_title("Top 10 Products by Revenue (£k)")
        axes[0, 1].set_xlabel("Revenue (£k)")

        # Top countries ex-UK
        top_c = (df[df["Country"] != "United Kingdom"]
                   .groupby("Country")["TotalPrice"].sum()
                   .sort_values(ascending=False).head(10))
        axes[1, 0].bar(top_c.index, top_c.values / 1e3, color=PALETTE[2])
        axes[1, 0].set_title("Top 10 Countries ex-UK (£k)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Spend distribution
        spend = df.groupby("CustomerID")["TotalPrice"].sum()
        axes[1, 1].hist(spend[spend < spend.quantile(0.95)],
                        bins=50, color=PALETTE[3], edgecolor="white")
        axes[1, 1].set_title("Customer Spend Distribution (95th pct)")
        axes[1, 1].set_xlabel("Total Spend (£)")

        plt.tight_layout()
        return self._save(fig, "eda_dashboard.png")

    # ── Segmentation ───────────────────────────────

    def cluster_selection(self, elbow_data: dict) -> plt.Figure:
        """Elbow + silhouette score plots for k selection."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("K-Means Cluster Selection", fontsize=14)

        axes[0].plot(elbow_data["k"], elbow_data["inertia"],
                     marker="o", color=PALETTE[0])
        axes[0].set_title("Elbow Method")
        axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")

        axes[1].plot(elbow_data["k"], elbow_data["silhouette"],
                     marker="s", color=PALETTE[3])
        axes[1].set_title("Silhouette Score")
        axes[1].set_xlabel("k")

        plt.tight_layout()
        return self._save(fig, "cluster_selection.png")

    def customer_segments(self, rfm: pd.DataFrame) -> plt.Figure:
        """Scatter plot coloured by cluster."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(
            rfm["Recency"], np.log1p(rfm["Monetary"]),
            c=rfm["Cluster"], cmap="viridis", alpha=0.5, s=20,
        )
        ax.set_title("Customer Segments: Recency vs Log(Monetary)")
        ax.set_xlabel("Recency (days)")
        ax.set_ylabel("Log(Total Spend £)")
        plt.colorbar(sc, label="Cluster")
        plt.tight_layout()
        return self._save(fig, "customer_segments.png")

    def segment_revenue_share(self, rfm: pd.DataFrame) -> plt.Figure:
        """Pie chart of revenue share per segment."""
        rev = rfm.groupby("Segment")["Monetary"].sum()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(rev, labels=rev.index, autopct="%1.1f%%",
               colors=PALETTE, startangle=140)
        ax.set_title("Revenue Share by Segment")
        plt.tight_layout()
        return self._save(fig, "segment_revenue_share.png")

    # ── Churn ──────────────────────────────────────

    def churn_model_comparison(self, results_df: pd.DataFrame) -> plt.Figure:
        """Horizontal bar chart comparing F1 scores across models."""
        fig, ax = plt.subplots(figsize=(8, 4))
        results_df["F1"].sort_values().plot(kind="barh", ax=ax, color=PALETTE[1])
        ax.set_title("Churn Prediction — F1 Score Comparison")
        ax.set_xlabel("F1 Score")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        return self._save(fig, "churn_model_comparison.png")

    def churn_feature_importance(self, importances: pd.Series) -> plt.Figure:
        """Feature importance bar chart for the churn model."""
        fig, ax = plt.subplots(figsize=(8, 4))
        importances.sort_values().plot(kind="barh", ax=ax, color=PALETTE[4])
        ax.set_title("Churn Model — Feature Importances")
        plt.tight_layout()
        return self._save(fig, "churn_feature_importance.png")

    def confusion_matrix_plot(self, cm: np.ndarray) -> plt.Figure:
        """Heatmap of the confusion matrix."""
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Active", "Churned"],
                    yticklabels=["Active", "Churned"])
        ax.set_title("Confusion Matrix — Churn")
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        plt.tight_layout()
        return self._save(fig, "confusion_matrix.png")

    # ── CLV ────────────────────────────────────────

    def clv_actual_vs_predicted(self, residuals_df: pd.DataFrame) -> plt.Figure:
        """Scatter of actual vs predicted CLV (log scale)."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(residuals_df["actual_log"], residuals_df["predicted_log"],
                   alpha=0.3, color=PALETTE[0], s=15)
        lims = [residuals_df["actual_log"].min(), residuals_df["actual_log"].max()]
        ax.plot(lims, lims, "r--", lw=2, label="Perfect fit")
        ax.set_title("CLV Prediction: Actual vs Predicted (log scale)")
        ax.set_xlabel("Actual log(Monetary)")
        ax.set_ylabel("Predicted log(Monetary)")
        ax.legend()
        plt.tight_layout()
        return self._save(fig, "clv_actual_vs_predicted.png")

    def clv_feature_importance(self, importances: pd.Series) -> plt.Figure:
        """Feature importances for the CLV model."""
        fig, ax = plt.subplots(figsize=(8, 4))
        importances.sort_values().plot(kind="barh", ax=ax, color=PALETTE[2])
        ax.set_title("CLV Model — Feature Importances")
        plt.tight_layout()
        return self._save(fig, "clv_feature_importance.png")

    # ── Business Insights ──────────────────────────

    def pareto_curve(self, rfm: pd.DataFrame) -> plt.Figure:
        """Cumulative customer vs revenue Pareto curve."""
        rfm_sorted = rfm.sort_values("Monetary", ascending=False).copy()
        total_rev  = rfm["Monetary"].sum()
        rfm_sorted["CumRevPct"]  = rfm_sorted["Monetary"].cumsum() / total_rev * 100
        rfm_sorted["CumCustPct"] = np.arange(1, len(rfm_sorted) + 1) / len(rfm_sorted) * 100

        pct_80 = rfm_sorted.loc[rfm_sorted["CumRevPct"] <= 80, "CumCustPct"].max()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(rfm_sorted["CumCustPct"], rfm_sorted["CumRevPct"],
                color=PALETTE[0], lw=2)
        ax.axhline(80, color="red", linestyle="--", alpha=0.7, label="80% revenue")
        ax.axvline(pct_80, color="green", linestyle="--", alpha=0.7,
                   label=f"{pct_80:.0f}% customers")
        ax.fill_between(rfm_sorted["CumCustPct"], rfm_sorted["CumRevPct"],
                        alpha=0.08, color=PALETTE[0])
        ax.set_title("Pareto: Customer Revenue Concentration")
        ax.set_xlabel("Cumulative % Customers")
        ax.set_ylabel("Cumulative % Revenue")
        ax.legend(); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        plt.tight_layout()
        return self._save(fig, "pareto_curve.png")

    # ── Helper ─────────────────────────────────────

    def _save(self, fig: plt.Figure, filename: str) -> plt.Figure:
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        print(f"[visualize] Saved → {path}")
        plt.close(fig)
        return fig
