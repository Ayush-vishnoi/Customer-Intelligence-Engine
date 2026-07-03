"""
app.py — Flask server for E-Commerce Customer Intelligence System
Routes:
  GET  /                  → Dashboard (analytics overview)
  GET  /predict           → Prediction tool page
  POST /api/predict       → Single-customer prediction (JSON)
  POST /api/batch         → Batch CSV prediction (JSON)
  GET  /api/segments      → Segment data for charts
  GET  /api/revenue       → Monthly revenue data
  GET  /api/churn-stats   → Churn statistics
  GET  /api/clv-stats     → CLV statistics
  GET  /api/top-products  → Top products data
  GET  /api/customers     → Paginated customer table
"""

import os, sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import Predictor

# ── App setup ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(BASE_DIR, "..")
DATA_DIR   = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

app        = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

from whitenoise import WhiteNoise
app.wsgi_app = WhiteNoise(app.wsgi_app, root=os.path.join(BASE_DIR, "static"), prefix="static")
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor(models_dir=MODELS_DIR)
    return _predictor

# ── Cached CSV loaders ────────────────────────────────────────────────────────
_cache = {}

def _csv(filename):
    if filename not in _cache:
        _cache[filename] = pd.read_csv(os.path.join(DATA_DIR, filename))
    return _cache[filename].copy()

def get_rfm():      return _csv("customer_segments.csv")
def get_monthly():  return _csv("monthly_revenue.csv")
def get_products(): return _csv("top_products.csv")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/customers")
def customers_page():
    return render_template("customers.html")


# ════════════════════════════════════════════════════════════════════════════════
# API ROUTES — Analytics
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/api/kpis")
def api_kpis():
    rfm = get_rfm()
    total_rev   = float(rfm["Monetary"].sum())
    n_customers = int(len(rfm))
    churn_rate  = float(rfm["Churned"].mean() * 100)
    avg_clv     = float(rfm["PredictedCLV_GBP"].mean()) if "PredictedCLV_GBP" in rfm.columns else 0.0

    rfm_s = rfm.sort_values("Monetary", ascending=False)
    rfm_s["cumrev"] = rfm_s["Monetary"].cumsum() / total_rev * 100
    rfm_s["cumcust"] = np.arange(1, len(rfm_s)+1) / len(rfm_s) * 100
    top_pct = float(rfm_s[rfm_s["cumrev"] <= 80]["cumcust"].max())

    q75 = rfm["Frequency"].quantile(0.75)
    loyalty_mult = float(
        rfm[rfm["Frequency"] >= q75]["Monetary"].mean() /
        rfm[rfm["Frequency"] <  q75]["Monetary"].mean()
    )
    return jsonify({
        "total_revenue":   total_rev,
        "n_customers":     n_customers,
        "churn_rate":      round(churn_rate, 1),
        "avg_clv":         round(avg_clv, 2),
        "top_pct_80":      round(top_pct, 1),
        "loyalty_mult":    round(loyalty_mult, 1),
    })


@app.route("/api/segments")
def api_segments():
    rfm = get_rfm()
    total = rfm["Monetary"].sum()

    seg_data = (
        rfm.groupby("Segment").agg(
            count      = ("CustomerID",      "count"),
            revenue    = ("Monetary",        "sum"),
            avg_spend  = ("Monetary",        "mean"),
            avg_rec    = ("Recency",         "mean"),
            avg_freq   = ("Frequency",       "mean"),
            churn_rate = ("Churned",         "mean"),
        ).reset_index()
    )
    seg_data["revenue_pct"] = (seg_data["revenue"] / total * 100).round(1)
    seg_data = seg_data.round(2)
    return jsonify(seg_data.to_dict(orient="records"))


@app.route("/api/revenue-monthly")
def api_revenue_monthly():
    monthly = get_monthly()
    return jsonify({"labels": monthly["InvoiceDate"].tolist(), "values": monthly["TotalPrice"].round(2).tolist()})


@app.route("/api/churn-stats")
def api_churn_stats():
    rfm = get_rfm()
    bands = [0, 0.3, 0.6, 1.01]
    labels = ["Low (<30%)", "Medium (30-60%)", "High (>60%)"]
    if "ChurnProbability" in rfm.columns:
        counts = pd.cut(rfm["ChurnProbability"], bins=bands, labels=labels).value_counts()
        return jsonify({
            "labels": labels,
            "counts": [int(counts.get(l, 0)) for l in labels],
            "overall_rate": round(float(rfm["Churned"].mean() * 100), 1),
        })
    counts_raw = rfm["Churned"].value_counts()
    return jsonify({
        "labels": ["Active", "Churned"],
        "counts": [int(counts_raw.get(0, 0)), int(counts_raw.get(1, 0))],
        "overall_rate": round(float(rfm["Churned"].mean() * 100), 1),
    })


@app.route("/api/clv-distribution")
def api_clv_distribution():
    rfm = get_rfm()
    if "PredictedCLV_GBP" not in rfm.columns:
        return jsonify({"bins": [], "counts": []})
    clv = rfm["PredictedCLV_GBP"]
    p95 = clv.quantile(0.95)
    hist, edges = np.histogram(clv[clv <= p95], bins=30)
    return jsonify({
        "bins":   [round(float(e), 2) for e in edges[:-1]],
        "counts": hist.tolist(),
        "mean":   round(float(clv.mean()), 2),
        "median": round(float(clv.median()), 2),
        "p90":    round(float(clv.quantile(0.9)), 2),
    })


@app.route("/api/pareto")
def api_pareto():
    rfm = get_rfm()
    rfm_s = rfm.sort_values("Monetary", ascending=False).copy()
    total = rfm_s["Monetary"].sum()
    rfm_s["CumRevPct"]  = (rfm_s["Monetary"].cumsum() / total * 100).round(2)
    rfm_s["CumCustPct"] = (np.arange(1, len(rfm_s)+1) / len(rfm_s) * 100).round(2)
    # downsample to 200 points for frontend
    step = max(1, len(rfm_s) // 200)
    sample = rfm_s.iloc[::step]
    return jsonify({
        "cust_pct": sample["CumCustPct"].tolist(),
        "rev_pct":  sample["CumRevPct"].tolist(),
    })


@app.route("/api/top-products")
def api_top_products():
    return jsonify(get_products().to_dict(orient="records"))


@app.route("/api/customers")
def api_customers():
    rfm = get_rfm()
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    segment  = request.args.get("segment", "all")
    sort_by  = request.args.get("sort", "Monetary")
    order    = request.args.get("order", "desc")

    if segment != "all" and "Segment" in rfm.columns:
        rfm = rfm[rfm["Segment"] == segment]

    if sort_by in rfm.columns:
        rfm = rfm.sort_values(sort_by, ascending=(order == "asc"))

    total  = len(rfm)
    start  = (page - 1) * per_page
    subset = rfm.iloc[start:start + per_page]

    cols = ["CustomerID", "Recency", "Frequency", "Monetary",
            "Segment", "Churned"]
    if "ChurnProbability" in rfm.columns:
        cols.append("ChurnProbability")
    if "PredictedCLV_GBP" in rfm.columns:
        cols.append("PredictedCLV_GBP")

    present = [c for c in cols if c in subset.columns]
    records = subset[present].round(2).to_dict(orient="records")

    return jsonify({"total": total, "page": page,
                    "per_page": per_page, "data": records})


# ════════════════════════════════════════════════════════════════════════════════
# API ROUTES — Prediction
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    try:
        result = get_predictor().predict(
            recency   = float(data["recency"]),
            frequency = float(data["frequency"]),
            monetary  = float(data["monetary"]),
        )
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/batch", methods=["POST"])
def api_batch():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files["file"]
    try:
        df = pd.read_csv(file)
        required = {"CustomerID", "Recency", "Frequency", "Monetary"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            return jsonify({"success": False,
                            "error": f"Missing columns: {missing}"}), 400
        results = get_predictor().batch_predict(df)
        return jsonify({
            "success": True,
            "count":   len(results),
            "data":    results.round(2).to_dict(orient="records"),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, port=port, use_reloader=False)
