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
    df[num_cols] = df[num_cols].fillna(0)

    for c in TE_cols:
        df[c] = df[c].fillna("Unknown").astype(str)

    # Feature engineering (giống training)
    df["rev_log"] = np.log1p(df["expected_revenue"])
    df["rev_per_day"] = df["expected_revenue"] / (df["lead_age_days"] + 1)

    # Nếu không có create_date từ client → force default
    if "create_date" in df:
        df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce")
        df["create_month"] = df["create_date"].dt.month.fillna(0)
        df["create_dow"] = df["create_date"].dt.dayofweek.fillna(0)
    else:
        df["create_month"] = 0
        df["create_dow"] = 0

    # Target encoding
    df[TE_cols] = encoder.transform(df[TE_cols])

    # Đảm bảo đúng thứ tự cột features dùng khi train
    feature_cols = (
        num_cols
        + TE_cols
        + ["rev_log", "rev_per_day", "create_month", "create_dow"]
    )

    X = df[feature_cols]

    prob = model.predict_proba(X)[0][1]
    label = int(prob >= 0.5)

    return {
        "predicted_prob": float(prob),
        "predicted_label": label
    }
