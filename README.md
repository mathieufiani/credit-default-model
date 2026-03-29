# Credit Default Risk — ML Scoring Pipeline

A production-grade credit scoring system built on 300K+ consumer loan applications. Predicts payment difficulties and explains decisions using SHAP — replicating how AI-driven underwriting works at companies like Pagaya.

**Kaggle Leaderboard Score: 0.771 AUC** (Top ~15% of submissions)

---

## Problem

Consumer lenders face a critical challenge: approving borrowers who will default leads to direct financial losses, while rejecting creditworthy borrowers means missed revenue. Traditional FICO-based underwriting leaves money on the table by over-relying on a single score.

This project builds an ML pipeline that:
- Predicts the probability of payment difficulty for any loan applicant
- Explains *why* a borrower is flagged as risky — a regulatory requirement in consumer lending
- Serves predictions via a REST API, ready for integration into an underwriting system

---

## Results

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.777 |
| Gini Coefficient | 0.554 |
| KS Statistic | 0.410 |
| Kaggle Public Score | 0.771 |

**Iteration history:**

| Version | Kaggle Score |
|---------|-------------|
| XGBoost baseline (manual preprocessing) | 0.715 |
| sklearn Pipeline (clean preprocessing) | 0.763 |
| + Bureau credit history features | 0.768 |
| + Previous application features | 0.771 |
| + Optuna hyperparameter optimization | 0.771 |

---

## Key Findings

Top drivers of payment difficulty (via SHAP):

1. **External credit scores** (EXT_SOURCE_1/2/3) — the strongest predictors. Borrowers with low external scores are significantly more likely to default
2. **Previous application refusal rate** — borrowers previously rejected by lenders are high-risk signals
3. **Age** (DAYS_BIRTH) — younger borrowers default at higher rates (~10% for under-30 vs ~6% for over-45)
4. **Employment stability** (DAYS_EMPLOYED) — shorter employment history correlates strongly with default
5. **Debt burden** (CREDIT_TERM, ANNUITY_INCOME_RATIO) — high annuity-to-income ratios increase risk

---

## Architecture

```
Raw Data (307K loans)
    │
    ├── application_train.csv   (main features)
    ├── bureau.csv              (external credit history)
    └── previous_application.csv (past loan applications)
    │
    ▼
sklearn Pipeline
    ├── Median imputation (numerical)
    ├── Unknown imputation + OneHotEncoding (categorical)
    └── XGBoost Classifier (scale_pos_weight for class imbalance)
    │
    ▼
Evaluation: AUC-ROC, KS Statistic, Gini Coefficient
    │
    ▼
SHAP Explainability
    ├── Beeswarm plot (global feature importance)
    └── Waterfall plot (individual loan explanation)
    │
    ▼
FastAPI scoring endpoint
    POST /score → { default_probability, risk_label, top_3_reasons }
```

---

## API

Start the server:
```bash
uvicorn app:app --reload
```

Score a borrower:
```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_INCOME_TOTAL": 150000,
    "AMT_CREDIT": 200000,
    "AMT_ANNUITY": 10000,
    "DAYS_BIRTH": -18000,
    "DAYS_EMPLOYED": -3000,
    "EXT_SOURCE_1": 0.8,
    "EXT_SOURCE_2": 0.85,
    "EXT_SOURCE_3": 0.9,
    "CODE_GENDER": "F",
    "PREV_REFUSAL_RATE": 0.0
  }'
```

Response:
```json
{
  "default_probability": 0.06,
  "risk_label": "low",
  "top_3_reasons": [
    {"feature": "EXT_SOURCE_2", "shap_value": -1.02},
    {"feature": "EXT_SOURCE_3", "shap_value": -0.80},
    {"feature": "EXT_SOURCE_1", "shap_value": -0.55}
  ]
}
```

---

## Setup

```bash
git clone https://github.com/mathieufiani/credit-default-model
cd credit-default-model
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Data:** Download from [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) on Kaggle. Place `application_train.csv`, `bureau.csv`, and `previous_application.csv` in the root directory.

**Run notebooks in order:**
1. `01_eda.ipynb` — EDA & feature engineering
2. `02_modeling.ipynb` — XGBoost baseline + Optuna
3. `03_shap.ipynb` — SHAP explainability
4. `04_submission.ipynb` — Kaggle submission
5. `05_pipeline.ipynb` — Production pipeline + bureau/previous features

---

## Stack

| Tool | Usage |
|------|-------|
| XGBoost | Credit scoring model |
| SHAP | Explainability (regulatory requirement) |
| sklearn Pipeline | Clean, reproducible preprocessing |
| Optuna | Hyperparameter optimization |
| FastAPI | Scoring API |
| MLflow | Experiment tracking |

---

## About

Built as part of a credit risk portfolio ahead of a Portfolio Risk Analytics internship at Pagaya (May 2025). Designed to replicate real-world underwriting AI pipelines.

**Contact:** mathieu.fiani@gmail.com | [LinkedIn](https://linkedin.com/in/mathieu-fiani)
