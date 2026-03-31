# E-Commerce Customer Intelligence System

> End-to-end ML pipeline: customer segmentation, churn prediction, and lifetime value estimation on the UCI Online Retail dataset.

---

## What This Project Does

| Task | Method | Result |
|---|---|---|
| Customer segmentation | K-Means (RFM features) | 3 segments: High-Value, Loyal, At-Risk |
| Churn prediction | Random Forest classifier | F1 = 0.53, Accuracy = 71% |
| CLV regression | Random Forest regressor | R² = 0.91 |
| Business insights | Pareto + cohort analysis | 26% of customers → 80% of revenue |

---

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/your-username/ecommerce-intelligence.git
cd ecommerce-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Place Online_Retail.xlsx in data/

# 4. Run the full pipeline
python main.py

# 5. Launch the Streamlit app
streamlit run app/app.py
```

---

## Project Structure

```
ecommerce_intelligence/
├── data/
│   ├── Online_Retail.xlsx          ← raw dataset (not in git)
│   └── processed/
│       ├── rfm_features.csv
│       └── customer_segments.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_segmentation.ipynb
│   ├── 04_churn_prediction.ipynb
│   └── 05_clv_regression.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py      ← load & clean raw data
│   ├── features.py         ← RFM + log transforms
│   ├── segmentation.py     ← K-Means clustering
│   ├── churn_model.py      ← classifier training & evaluation
│   ├── clv_model.py        ← regression training & evaluation
│   └── visualize.py        ← all charts
│
├── models/
│   ├── kmeans_model.pkl
│   ├── churn_rf_model.pkl
│   └── clv_rf_model.pkl
│
├── reports/
│   ├── figures/            ← all generated .png charts
│   └── insights_summary.md
│
├── app/
│   ├── app.py              ← Streamlit entry point
│   └── predictor.py        ← model inference wrapper
│
├── main.py                 ← run full pipeline
├── requirements.txt
└── .gitignore
```

---

## Key Business Insights

- **26%** of customers generate **80%** of revenue (stronger than classic 80/20)
- **33.4%** of customers are at risk of churning (inactive > 90 days)
- High-frequency buyers spend **7.2×** more than occasional buyers
- The **High-Value** segment drives **75%** of total revenue

---

## Dataset

[UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)
— UK-based online retail, Dec 2010 to Dec 2011, 541K transactions.

---

## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `streamlit` · `joblib`

---

## Author

**Your Name** — [github.com/your-username](https://github.com/your-username)
