from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import shap
import json

app = FastAPI()

# Charger le modèle et les références
with open('final_model.pkl', 'rb') as f:
    final_model = pickle.load(f)

X_train = pd.read_csv('data/clean/X_train.csv')
train_columns = X_train.columns.tolist()

explainer = shap.Explainer(final_model)

class Applicant(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    EXT_SOURCE_1: float = 0.5
    EXT_SOURCE_2: float = 0.5
    EXT_SOURCE_3: float = 0.5
    CODE_GENDER: str = "M"

@app.post("/score")
def score(applicant: Applicant):
    # Créer les features dérivées
    data = {
        'AMT_INCOME_TOTAL': applicant.AMT_INCOME_TOTAL,
        'AMT_CREDIT': applicant.AMT_CREDIT,
        'AMT_ANNUITY': applicant.AMT_ANNUITY,
        'DAYS_BIRTH': applicant.DAYS_BIRTH,
        'DAYS_EMPLOYED': applicant.DAYS_EMPLOYED,
        'EXT_SOURCE_1': applicant.EXT_SOURCE_1,
        'EXT_SOURCE_2': applicant.EXT_SOURCE_2,
        'EXT_SOURCE_3': applicant.EXT_SOURCE_3,
        'CODE_GENDER_M': 1 if applicant.CODE_GENDER == 'M' else 0,
        'CREDIT_INCOME_RATIO': applicant.AMT_CREDIT / applicant.AMT_INCOME_TOTAL,
        'ANNUITY_INCOME_RATIO': applicant.AMT_ANNUITY / applicant.AMT_INCOME_TOTAL,
        'EMPLOYED_TO_AGE_RATIO': applicant.DAYS_EMPLOYED / applicant.DAYS_BIRTH,
        'CREDIT_TERM': applicant.AMT_ANNUITY / applicant.AMT_CREDIT,
        'INCOME_PER_PERSON': applicant.AMT_INCOME_TOTAL,
    }

    # Aligner avec les colonnes du train
    df = pd.DataFrame([data])
    df = df.reindex(columns=train_columns, fill_value=0)

    # Prédire
    proba = final_model.predict_proba(df)[0][1]

    # SHAP values
    shap_values = explainer(df)
    shap_dict = dict(zip(train_columns, shap_values.values[0]))
    top_3 = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    # Risk label
    if proba < 0.1:
        risk_label = "low"
    elif proba < 0.3:
        risk_label = "medium"
    else:
        risk_label = "high"

    return {
        "default_probability": round(float(proba), 4),
        "risk_label": risk_label,
        "top_3_reasons": [
            {"feature": f, "shap_value": round(float(v), 4)}
            for f, v in top_3
        ]
    }