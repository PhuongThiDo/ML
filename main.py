from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("lead_win_model_xgb.pkl")

# Danh sách cột đúng theo model đã train
MODEL_FEATURES = [
    "lead_age_days",
    "expected_revenue",
    "probability",
    "stage_id",
    "stage_sequence",
    "source_id",
    "campaign_id",
    "salesperson_id",
    "team_id",
    "customer_region",
    "priority",
    "create_month",
    "create_dayofweek",
    "log_expected_revenue",
    "rev_per_day_age"
]

@app.get("/")
def root():
    return {"status": "API OK - Model loaded"}

@app.post("/predict")
def predict(raw: dict):

    # Convert input → DataFrame
    df = pd.DataFrame([raw])

    # Tính lại các feature engineered (giống dataset train)
    if "expected_revenue" in raw:
        df["log_expected_revenue"] = df["expected_revenue"].apply(lambda x: 0 if x <= 0 else pd.np.log(x))
    else:
        df["log_expected_revenue"] = 0

    if "expected_revenue" in raw and "lead_age_days" in raw:
        df["rev_per_day_age"] = df.apply(
            lambda r: (r["expected_revenue"] / r["lead_age_days"]) if r["lead_age_days"] > 0 else 0,
            axis=1
        )
    else:
        df["rev_per_day_age"] = 0

    # Ensure all columns exist
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = None

    # Reorder columns
    df = df[MODEL_FEATURES]

    # Predict
    prob = model.predict_proba(df)[0][1]
    label = int(prob >= 0.5)

    return {
        "predicted_prob": float(prob),
        "predicted_label": label
    }
