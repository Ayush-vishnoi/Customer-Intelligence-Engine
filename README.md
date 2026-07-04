# Customer Intelligence Engine

> End-to-end ML pipeline for customer segmentation, churn prediction, and lifetime value estimation — served via a Flask web app and deployed on Render.

🔗 **Live Demo:** [customer-intelligence-engine-e9jl.onrender.com](https://customer-intelligence-engine-e9jl.onrender.com)

Built on the [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail): 541K transactions, 4,338 customers, UK-based e-commerce, Dec 2010 – Dec 2011.

---

## What It Does

| Task | Method | Result |
|---|---|---|
| Customer segmentation | K-Means on RFM features | 3 segments: High-Value 💎, Loyal 🌟, At-Risk ⚠️ |
| Churn prediction | Random Forest classifier | F1 = 0.53, Accuracy = 71% |
| CLV regression | Random Forest regressor | R² = 0.91 |
| Business insights | Pareto + cohort analysis | 26% of customers → 80% of revenue |

---

## Key Business Insights

- **26%** of customers generate **80%** of revenue — stronger than the classic 80/20 rule
- **33.4%** of customers are at risk of churning (inactive > 90 days)
- High-frequency buyers spend **7.2×** more than occasional buyers
- The **High-Value** segment drives **75%** of total revenue

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/your-username/customer-intelligence-engine.git
cd customer-intelligence-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the raw dataset
# Place Online_Retail.xlsx in data/
# Download from: https://archive.ics.uci.edu/ml/datasets/online+retail

# 4. Run the full ML pipeline
# Trains all models, generates processed CSVs and report charts
python main.py

# 5. Launch the web app
python run.py

# Open: http://localhost:5050
```

> If port 5050 is taken: `python run.py --port 8888`
> For production mode: `python run.py --prod`

---

## Web App — Pages

| Page | URL | Description |
|---|---|---|
| Dashboard | `/` | KPI cards, 6 interactive charts, segment table |
| Predictor | `/predict` | Single customer + batch CSV prediction |
| Customers | `/customers` | Browse all 4,338 customers with filters & sorting |

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/api/kpis` | Total revenue, churn rate, avg CLV, Pareto %, loyalty multiplier |
| GET | `/api/segments` | Segment breakdown with revenue share and avg spend |
| GET | `/api/revenue-monthly` | Monthly revenue time series |
| GET | `/api/churn-stats` | Churn risk band distribution (Low / Medium / High) |
| GET | `/api/clv-distribution` | CLV histogram with mean, median, p90 |
| GET | `/api/pareto` | Pareto curve data points |
| GET | `/api/top-products` | Top 10 products by revenue |
| GET | `/api/customers` | Paginated, filterable customer list |
| POST | `/api/predict` | Single customer prediction (JSON body) |
| POST | `/api/batch` | Batch CSV prediction (file upload) |

### Example — Single Prediction

```bash
curl -X POST https://customer-intelligence-engine-e9jl.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"recency": 15, "frequency": 8, "monetary": 3200}'
```

```json
{
  "success": true,
  "result": {
    "segment": "High-Value",
    "churn_probability": 0.0821,
    "churn_risk": "Low",
    "predicted_clv_gbp": 4150.30
  }
}
```

### Example — Batch Prediction

Upload a CSV with columns `CustomerID`, `Recency`, `Frequency`, `Monetary`:

```bash
curl -X POST https://customer-intelligence-engine-e9jl.onrender.com/api/batch \
  -F "file=@customers.csv"
```

---

## ML Pipeline (`main.py`)

Runs 7 steps end-to-end:

| Step | What happens |
|---|---|
| 1. Data loading | Reads `Online_Retail.xlsx`, drops nulls, cancellations, invalid rows |
| 2. EDA | Generates 4-panel dashboard chart (revenue, products, countries, spend dist.) |
| 3. Feature engineering | Builds RFM table + log transforms + churn label (90-day threshold) |
| 4. Segmentation | K-Means with auto k-selection via silhouette score → 3 clusters |
| 5. Churn prediction | Trains Logistic Regression + Random Forest, picks best by F1 |
| 6. CLV regression | Trains Linear + Random Forest on log(Monetary), picks best by R² |
| 7. Business insights | Pareto curve, segment revenue share, loyalty multiplier |

**Outputs:**
- `models/` — 3 trained `.pkl` files
- `data/processed/` — `customer_segments.csv`, `monthly_revenue.csv`, `top_products.csv`
- `reports/figures/` — 10 chart PNGs

---

## Project Structure

```
customer-intelligence-engine/
├── app/
│   ├── static/
│   │   ├── css/main.css
│   │   ├── img/favicon.png
│   │   └── js/
│   │       ├── chart.umd.min.js
│   │       ├── main.js          ← shared utilities & formatters
│   │       ├── dashboard.js     ← dashboard charts & KPIs
│   │       ├── predict.js       ← prediction form logic
│   │       └── customers.js     ← customer table logic
│   ├── templates/
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── predict.html
│   │   └── customers.html
│   ├── app.py                   ← Flask routes & API
│   └── predictor.py             ← model inference wrapper
│
├── data/
│   ├── Online_Retail.xlsx       ← raw dataset (not in git)
│   └── processed/
│       ├── customer_segments.csv
│       ├── monthly_revenue.csv
│       └── top_products.csv
│
├── models/
│   ├── kmeans_model.pkl
│   ├── churn_rf_model.pkl
│   └── clv_rf_model.pkl
│
├── src/
│   ├── data_loader.py           ← load & clean raw data
│   ├── features.py              ← RFM + log transforms + churn label
│   ├── segmentation.py          ← K-Means clustering
│   ├── churn_model.py           ← classifier training & evaluation
│   ├── clv_model.py             ← regression training & evaluation
│   └── visualize.py             ← all report charts
│
├── reports/
│   ├── figures/                 ← generated PNGs (not in git)
│   └── insights_summary.md
│
├── main.py                      ← run full ML pipeline
├── run.py                       ← launch Flask locally
├── Procfile                     ← Render/gunicorn start command
├── render.yaml                  ← Render deployment config
├── requirements.txt
└── .gitignore
```

---

## Deploying to Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → Web Service → connect repo
3. Render auto-detects `render.yaml` — no manual config needed

**Start command:**
```
gunicorn "app.app:app" --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

> Static files are served via [WhiteNoise](https://whitenoise.readthedocs.io/) — no separate CDN needed.

---

## Customer Segments

| Segment | Behaviour | Median Recency | Median Spend | Revenue Share |
|---|---|---|---|---|
| High-Value 💎 | Frequent, high-spend, recent | ~15 days | £3,481 | ~75% |
| Loyal 🌟 | Regular, moderate spend | ~40 days | £627 | ~20% |
| At-Risk ⚠️ | Infrequent, long inactive | ~246 days | £293 | ~5% |

**Recommended actions:**
- **High-Value** — VIP service, early product access, dedicated support
- **Loyal** — Upsell/cross-sell campaigns, reward points
- **At-Risk** — Time-limited win-back discount, re-engagement email

---

## Model Performance

### Churn Prediction

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.717 | 0.609 | 0.424 | 0.500 |
| Random Forest ✓ | 0.713 | 0.586 | 0.483 | **0.529** |

### CLV Regression

| Model | RMSE (log) | R² |
|---|---|---|
| Linear Regression | 0.420 | 0.890 |
| Random Forest ✓ | **0.385** | **0.908** |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | `pandas` · `numpy` · `scikit-learn` · `joblib` |
| Visualisation | `matplotlib` · `seaborn` |
| Web app | `flask` · `gunicorn` · `whitenoise` |
| Frontend | Chart.js · vanilla JS |
| Deployment | Render |

---

## Dataset

[UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)
— 541,909 transactions · 4,338 customers · 37 countries · Dec 2010 – Dec 2011

The raw `.xlsx` file is excluded from git (23MB). Download it from the UCI link above and place it at `data/Online_Retail.xlsx` before running `main.py`.
