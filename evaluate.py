import joblib
import math
import json
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split

# === 1. Ladda modell och metadata ===
model_data = joblib.load("app/model.joblib")
model = model_data["model"]
scaler = model_data["scaler"]
threshold = model_data.get("threshold", None)
version = model_data.get("version", "unknown")

# === 2. Ladda diabetes dataset ===
data = load_diabetes(as_frame=True)
X = data.frame.drop(columns=["target"])
y = data.frame["target"]

# === 3. Dela upp data (samma som i train.py) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 4. Skala testdata ===
X_test_scaled = scaler.transform(X_test)

# === 5. Prediktera ===
y_pred = model.predict(X_test_scaled)

# === 6. RMSE ===
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

# === 7. High-risk metrics ===
y_test_highrisk = (y_test > np.percentile(y_train, 75)).astype(int)
y_pred_highrisk = (y_pred > np.percentile(y_train, 75)).astype(int)
precision = precision_score(y_test_highrisk, y_pred_highrisk)
recall = recall_score(y_test_highrisk, y_pred_highrisk)

# === 8. Skriv ut resultat ===
print(f"Model version: {version}")
print(f"RMSE: {rmse:.2f}")
print(f"Precision (high-risk): {precision:.2f}")
print(f"Recall (high-risk): {recall:.2f}")

# === 9. Spara metrics till fil ===
metrics = {
    "version": version,
    "rmse": round(rmse, 2),
    "precision": round(precision, 2),
    "recall": round(recall, 2)
}
with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
