from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import shap

app = FastAPI()

# Charger le pipeline
with open('pipeline_final.pkl', 'rb') as f:
    final_pipeline = pickle.load(f)

# Charger un template avec toutes les colonnes
X_train = pd.read_csv('data/clean/X_train.csv')
template = X_train.median(numeric_only=True).to_dict()

# Valeurs par défaut pour les catégorielles
cat_defaults = {
    'NAME_CONTRACT_TYPE': 'Cash loans',
    'CODE_GENDER': 'F',
    'FLAG_OWN_CAR': 'N',
    'FLAG_OWN_REALTY': 'Y',
    'NAME_TYPE_SUITE': 'Unaccompanied',
    'NAME_INCOME_TYPE': 'Working',
    'NAME_EDUCATION_TYPE': 'Secondary / secondary special',
    'NAME_FAMILY_STATUS': 'Married',
    'NAME_HOUSING_TYPE': 'House / apartment',
    'OCCUPATION_TYPE': 'Unknown',
    'ORGANIZATION_TYPE': 'Unknown',
}
template.update(cat_defaults)

bureau_prev_defaults = {
    'BUREAU_CLOSED_COUNT': 0,
    'BUREAU_TOTAL_CREDIT': 0,
    'BUREAU_AVG_DAYS_CREDIT': 0,
    'BUREAU_ACTIVE_RATIO': 0,
    'PREV_APPROVED_COUNT': 0,
    'PREV_AMT_CREDIT_MEAN': 0,
    'PREV_AMT_ANNUITY_MEAN': 0,
    'PREV_DAYS_DECISION_MEAN': 0,
    'PREV_CNT_PAYMENT_MEAN': 0,
}
template.update(bureau_prev_defaults)

# Extraire les composants
preprocessor = final_pipeline.named_steps['preprocessor']
xgb_model = final_pipeline.named_steps['model']

# SHAP sur le modèle XGBoost uniquement
explainer = shap.TreeExplainer(xgb_model)

# Récupérer les noms de features après transformation
num_features = preprocessor.transformers_[0][2]
cat_features = preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names_out(
    preprocessor.transformers_[1][2]
).tolist()
all_features = num_features + cat_features

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
    BUREAU_LOAN_COUNT: int = 0
    BUREAU_ACTIVE_COUNT: int = 0
    BUREAU_BAD_DEBT_COUNT: int = 0
    BUREAU_MAX_OVERDUE: float = 0
    BUREAU_TOTAL_DEBT: float = 0
    PREV_APP_COUNT: int = 0
    PREV_REFUSED_COUNT: int = 0
    PREV_REFUSAL_RATE: float = 0

@app.post("/score")
def score(applicant: Applicant):
    data = {
        'AMT_INCOME_TOTAL': applicant.AMT_INCOME_TOTAL,
        'AMT_CREDIT': applicant.AMT_CREDIT,
        'AMT_ANNUITY': applicant.AMT_ANNUITY,
        'DAYS_BIRTH': applicant.DAYS_BIRTH,
        'DAYS_EMPLOYED': applicant.DAYS_EMPLOYED,
        'EXT_SOURCE_1': applicant.EXT_SOURCE_1,
        'EXT_SOURCE_2': applicant.EXT_SOURCE_2,
        'EXT_SOURCE_3': applicant.EXT_SOURCE_3,
        'CODE_GENDER': applicant.CODE_GENDER,
        'CREDIT_INCOME_RATIO': applicant.AMT_CREDIT / applicant.AMT_INCOME_TOTAL,
        'ANNUITY_INCOME_RATIO': applicant.AMT_ANNUITY / applicant.AMT_INCOME_TOTAL,
        'EMPLOYED_TO_AGE_RATIO': applicant.DAYS_EMPLOYED / applicant.DAYS_BIRTH,
        'CREDIT_TERM': applicant.AMT_ANNUITY / applicant.AMT_CREDIT,
        'INCOME_PER_PERSON': applicant.AMT_INCOME_TOTAL,
        'BUREAU_LOAN_COUNT': applicant.BUREAU_LOAN_COUNT,
        'BUREAU_ACTIVE_COUNT': applicant.BUREAU_ACTIVE_COUNT,
        'BUREAU_BAD_DEBT_COUNT': applicant.BUREAU_BAD_DEBT_COUNT,
        'BUREAU_MAX_OVERDUE': applicant.BUREAU_MAX_OVERDUE,
        'BUREAU_TOTAL_DEBT': applicant.BUREAU_TOTAL_DEBT,
        'PREV_APP_COUNT': applicant.PREV_APP_COUNT,
        'PREV_REFUSED_COUNT': applicant.PREV_REFUSED_COUNT,
        'PREV_REFUSAL_RATE': applicant.PREV_REFUSAL_RATE,
    }

    row = template.copy()
    row.update(data)
    df = pd.DataFrame([row])

    # Prédire via le pipeline complet
    proba = final_pipeline.predict_proba(df)[0][1]

    # SHAP via modèle seul, données transformées
    df_transformed = preprocessor.transform(df)
    shap_vals = explainer.shap_values(df_transformed)
    shap_dict = dict(zip(all_features, shap_vals[0]))
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