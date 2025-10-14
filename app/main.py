from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI(title="Diabetes Triage ML Service")

# === Ladda modell och metadata ===
model_data = joblib.load("app/model.joblib")
scaler = model_data["scaler"]
model = model_data["model"]
version = model_data.get("version", "unknown")
threshold = model_data.get("threshold", None)  # Tröskel för "high-risk"

# === Featurelista ===
FEATURES = [
    "age", "sex", "bmi", "bp", "s1",
    "s2", "s3", "s4", "s5", "s6"
]


@app.get("/health")
def health():
    """Hälsokontroll — visar att API:t körs."""
    return {"status": "ok", "model_version": version}


@app.post("/predict")
def predict(data: dict):
    """Tar emot patientdata och returnerar prediktion + risknivå."""
    try:
        df = pd.DataFrame([data])

        # Kontrollera att alla features finns
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing}"
            )

        # Kontrollera att värdena är numeriska
        for col in FEATURES:
            val = df[col].iloc[0]
            if not isinstance(val, (int, float)):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Feature '{col}' måste vara numerisk, "
                        f"fick: {val}"
                    )
                )

        # Skala och predicera
        X_scaled = scaler.transform(df[FEATURES])
        pred = model.predict(X_scaled)[0]

        # Bedöm high-risk baserat på tröskel
        risk_status = "Low/Normal risk"
        if threshold is not None and pred > threshold:
            risk_status = "High risk patient"

        # Returnera resultatet
        return {
            "prediction": round(float(pred), 2),
            "risk_assessment": risk_status
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
