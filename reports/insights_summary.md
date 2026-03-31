# Business Insights Summary

> Auto-generated after running `python main.py`

---

## Dataset Overview

| Metric | Value |
|---|---|
| Raw rows | 541,909 |
| Clean rows | ~392,692 |
| Unique customers | 4,338 |
| Countries | 37 |
| Date range | Dec 2010 – Dec 2011 |

---

## Key Findings

### 1. Pareto Effect (Revenue Concentration)
> **26% of customers generate 80% of revenue.**

This is even more concentrated than the classic 80/20 rule.
Focus retention efforts and personalisation on this top tier.

### 2. Churn Risk
> **33.4% of customers are inactive for > 90 days.**

One in three customers has likely churned. A win-back campaign
targeting customers in the 90–180 day window could yield significant recovery.

### 3. Loyalty Premium
> **High-frequency buyers spend 7.2× more than low-frequency buyers.**

Investing in loyalty programmes (points, early access, free shipping)
for repeat buyers has a measurable ROI.

---

## Customer Segments

| Segment | Median Recency | Median Frequency | Median Spend | Revenue Share |
|---|---|---|---|---|
| High-Value 💎 | ~15 days | ~8 orders | £3,481 | ~75% |
| Loyal 🌟 | ~40 days | ~2 orders | £627 | ~20% |
| At-Risk ⚠️ | ~246 days | ~1 order | £293 | ~5% |

### Recommended Actions per Segment

**High-Value** — Protect. Offer VIP service, early product access, dedicated support.

**Loyal** — Grow. Use targeted upsell / cross-sell campaigns, reward points.

**At-Risk** — Reactivate. Time-limited discount, personalised re-engagement email.

---

## Model Performance

### Churn Prediction

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.717 | 0.609 | 0.424 | 0.500 | — |
| Random Forest | 0.713 | 0.586 | 0.483 | **0.529** | — |

### CLV Regression

| Model | RMSE (log) | R² |
|---|---|---|
| Linear Regression | 0.420 | 0.890 |
| Random Forest | **0.385** | **0.908** |

The Random Forest regressor explains **90.8% of variance** in customer spend —
a strong result for a small feature set.

---

## Next Steps

- [ ] Add seasonality features (month of first purchase, holiday periods)
- [ ] Experiment with SMOTE to address churn class imbalance
- [ ] Deploy Streamlit app: `streamlit run app/app.py`
- [ ] Set up a retraining schedule (e.g. monthly) as new data arrives
