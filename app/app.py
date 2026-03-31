"""
app.py
------
Streamlit app for the E-Commerce Customer Intelligence System.

Run with:
    streamlit run app/app.py

Features
--------
- Single-customer prediction (segment, churn probability, CLV)
- Batch prediction via CSV upload
- Visual result cards with colour-coded risk level
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from predictor import Predictor

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="🛒",
    layout="centered",
)

SEGMENT_COLORS = {
    "High-Value": "#4361EE",
    "Loyal":      "#3A0CA3",
    "At-Risk":    "#F72585",
    "Low-Value":  "#888888",
}

# ── Cached model loader ───────────────────────────────────
@st.cache_resource
def get_predictor():
    return Predictor()


# ── Helpers ───────────────────────────────────────────────
def churn_badge(prob: float) -> str:
    if prob >= 0.7:
        return f"🔴 High risk ({prob*100:.0f}%)"
    elif prob >= 0.4:
        return f"🟡 Medium risk ({prob*100:.0f}%)"
    else:
        return f"🟢 Low risk ({prob*100:.0f}%)"


def render_result_card(result: dict):
    seg   = result["segment"]
    color = SEGMENT_COLORS.get(seg, "#4361EE")

    st.markdown(f"""
    <div style="
        border-left: 5px solid {color};
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    ">
        <h3 style="color:{color}; margin:0">Segment: {seg}</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Churn Risk", churn_badge(result["churn_probability"]))
    col2.metric("Predicted CLV", f"£{result['predicted_clv_gbp']:,.2f}")


# ── Main app ─────────────────────────────────────────────
def main():
    st.title("🛒 E-Commerce Customer Intelligence")
    st.markdown(
        "Predict a customer's **segment**, **churn probability**, and "
        "**lifetime value** using trained ML models."
    )

    predictor = get_predictor()

    tab1, tab2 = st.tabs(["Single Customer", "Batch Prediction"])

    # ── Tab 1: Single prediction ──────────────────
    with tab1:
        st.subheader("Enter Customer RFM Values")

        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.number_input(
                "Recency (days since last purchase)",
                min_value=1, max_value=730, value=45, step=1,
            )
        with col2:
            frequency = st.number_input(
                "Frequency (number of orders)",
                min_value=1, max_value=300, value=5, step=1,
            )
        with col3:
            monetary = st.number_input(
                "Monetary (total spend £)",
                min_value=1.0, max_value=500_000.0, value=800.0, step=50.0,
            )

        if st.button("Predict", type="primary"):
            with st.spinner("Running models..."):
                result = predictor.predict(recency, frequency, monetary)
            render_result_card(result)

            with st.expander("Raw output"):
                st.json(result)

    # ── Tab 2: Batch prediction ───────────────────
    with tab2:
        st.subheader("Upload a CSV for batch scoring")
        st.markdown(
            "CSV must have columns: **CustomerID**, **Recency**, "
            "**Frequency**, **Monetary**"
        )

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write(f"Loaded {len(df):,} rows")
            st.dataframe(df.head())

            required = {"CustomerID", "Recency", "Frequency", "Monetary"}
            if not required.issubset(df.columns):
                missing = required - set(df.columns)
                st.error(f"Missing columns: {missing}")
            else:
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner(f"Scoring {len(df):,} customers..."):
                        results_df = predictor.batch_predict(df)

                    st.success("Done!")
                    st.dataframe(results_df)

                    csv = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Results CSV",
                        data=csv,
                        file_name="customer_predictions.csv",
                        mime="text/csv",
                    )

    # ── Footer ────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Models: K-Means segmentation · Random Forest churn · "
        "Random Forest CLV regression · Dataset: UCI Online Retail"
    )


if __name__ == "__main__":
    main()
