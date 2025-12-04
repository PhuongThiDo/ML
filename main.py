from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load("lead_win_model_xgb.pkl")
encoder = joblib.load("encoder.pkl")

cat_cols = [
    "customer_region","source_id","campaign_id",
    "salesperson_id","team_id","stage_id","stage_sequence"
]

num_cols = ["expected_revenue","probability","lead_age_days","priority"]

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # ========== PREPROCESS giống lúc training ==========

    # numeric fill
    df[num_cols] = df[num_cols].fillna(0)

    # categorical fill + convert to string
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)

    # engineered features
    df["rev_log"] = np.log1p(df["expected_revenue"])
    df["rev_per_day"] = df["expected_revenue"] / (df["lead_age_days"] + 1)

    # create_date REQUIRED
    df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce")
    df["create_month"] = df["create_date"].dt.month
    df["create_dow"] = df["create_date"].dt.dayofweek

    # Target Encoding (giống training)
    df[cat_cols] = encoder.transform(df[cat_cols])

    # =====================================================

    prob = model.predict_proba(df)[0][1]
    label = int(prob >= 0.5)

    return {
        "predicted_prob": float(prob),
        "predicted_label": label
    }
