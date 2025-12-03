from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("lead_win_model_xgb.pkl")

@app.get("/")
def root():
    return {"status": "API OK - model loaded"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # Feature engineering (GIỐNG NHƯ LÚC TRAIN)
    df["log_expected_revenue"] = df["expected_revenue"].apply(
        lambda x: 0 if x <= 0 else np.log(x)
    )

    df["rev_per_day_age"] = df.apply(
        lambda r: (r["expected_revenue"] / r["lead_age_days"])
        if r["lead_age_days"] > 0 else 0,
        axis=1
    )

    # Predict
    prob = model.predict_proba(df)[0][1]
    label = int(prob >= 0.5)

    return {
        "predicted_prob": float(prob),
        "predicted_label": label
    }
