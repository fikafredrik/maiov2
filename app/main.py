from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI(title="Diabetes Triage ML Service")

# Load model
model_data = joblib.load("app/model.joblib")
scaler = model_data["scaler"]
model = model_data["model"]

# Field names
FEATURES = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
]


@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v0.1"}


@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        if not all(f in df.columns for f in FEATURES):
            missing = [f for f in FEATURES if f not in df.columns]
            raise HTTPException(
                status_code=400, detail=f"Missing features: {missing}"
            )
        X_scaled = scaler.transform(df[FEATURES])
        pred = model.predict(X_scaled)[0]
        return {"prediction": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
