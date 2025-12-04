from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load("lead_win_model_xgb.pkl")
encoder = joblib.load("encoder.pkl")

# Cột categorical dùng trong training
TE_cols = [
    "customer_region","source_id","campaign_id",
    "salesperson_id","team_id","stage_id","stage_sequence"
]

# Các cột numeric trong training
num_cols = ["expected_revenue","probability","lead_age_days","priority"]

@app.get("/")
def root():
    return {"status": "API OK - model loaded"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # Fill NA
    num_cols = ["expected_revenue","probability","lead_age_days","priority"]
    df[num_cols] = df[num_cols].fillna(0)

    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)

    # ========== FIX: Feature engineering giống 100% training ==========
    df["rev_log"] = np.log1p(df["expected_revenue"])
    df["rev_per_day"] = df["expected_revenue"] / (df["lead_age_days"] + 1)

    # NEW: thêm 3 features bị thiếu
    df["log_expected_revenue"] = np.log1p(df["expected_revenue"])
    df["rev_per_day_age"] = df["expected_revenue"] / (df["lead_age_days"] + 1)
    df["create_dayofweek"] = df["create_dow"]

    # Target encoding
    df[cat_cols] = encoder.transform(df[cat_cols])

    # ========= Giữ đúng thứ tự cột theo model ===============
    needed_columns = [
        "lead_age_days","expected_revenue","probability","stage_id",
        "stage_sequence","source_id","campaign_id","salesperson_id",
        "team_id","customer_region","priority","create_month",
        "create_dayofweek","log_expected_revenue","rev_per_day_age",
        "rev_log","rev_per_day","create_dow"
    ]
    df = df[needed_columns]

    # Predict
    prob = model.predict_proba(df)[0][1]
    label = int(prob >= 0.5)

    return {
        "predicted_prob": float(prob),
        "predicted_label": label
    }

