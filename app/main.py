from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI(title="Diabetes Triage ML Service")

# === 1. Ladda modellen ===
try:
    model_data = joblib.load("app/model.joblib")
    scaler = model_data["scaler"]
    model = model_data["model"]
    version = model_data.get("version", "unknown")
except Exception as e:
    # Om modellen inte går att ladda ska servern inte starta tyst
    raise RuntimeError(f"Kunde inte ladda modellen: {e}")

# === 2. Lista över alla features som modellen förväntar sig ===
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


# === 3. Health-check ===
@app.get("/health")
def health():
    """Returnerar status och aktuell modelversion."""
    return {"status": "ok", "model_version": version}


# === 4. Predict-endpoint ===
@app.post("/predict")
def predict(data: dict):
    """Tar emot JSON med feature-värden och returnerar en prediktion."""
    try:
        # Kontrollera att datan är en dictionary
        if not isinstance(data, dict):
            raise HTTPException(
                status_code=400,
                detail={"error": "Input must be a JSON object."}
            )

        # Kontrollera att alla features finns med
        missing = [f for f in FEATURES if f not in data]
        if missing:
            raise HTTPException(
                status_code=400,
                detail={"error": "Missing features.", "missing": missing}
            )

        # Konvertera till DataFrame
        df = pd.DataFrame([data])

        # Skala och förutsäg
        X_scaled = scaler.transform(df[FEATURES])
        pred = model.predict(X_scaled)[0]

        # Returnera resultatet som JSON
        return {"prediction": round(float(pred), 2), "model_version": version}

    except HTTPException:
        # Om vi redan kastat ett tydligt fel, låt det gå vidare
        raise
    except Exception as e:
        # Fångar andra typer av fel, t.ex. felaktig datatyp
        raise HTTPException(
            status_code=400,
            detail={"error": "Prediction failed.", "message": str(e)}
        )
