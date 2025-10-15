import random
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import joblib
import math
import json

# === Version & seeds (för reproducerbarhet) ===
version = "v0.2"
np.random.seed(42)
random.seed(42)

# === 1. Ladda data ===
data = load_diabetes(as_frame=True)
X = data.frame.drop(columns=["target"])
y = data.frame["target"]

# === 2. Dela upp data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 3. Skala data ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 4. Träna modell (Ridge Regression) ===
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# === 5. Prediktion och utvärdering ===
y_pred = model.predict(X_test_scaled)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE (v0.2 - Ridge): {rmse:.2f}")

# === 6. Skapa high-risk flagg ===
threshold = np.percentile(y_train, 75)
y_test_highrisk = (y_test > threshold).astype(int)
y_pred_highrisk = (y_pred > threshold).astype(int)

precision = precision_score(y_test_highrisk, y_pred_highrisk)
recall = recall_score(y_test_highrisk, y_pred_highrisk)

print(f"Precision (high-risk flag): {precision:.2f}")
print(f"Recall (high-risk flag): {recall:.2f}")

# === 7. Spara modell och scaler ===
model_artifact = {
    "model": model,
    "scaler": scaler,
    "threshold": threshold,
    "version": version
}
joblib.dump(model_artifact, "app/model.joblib")

# === 8. Logga metrics till JSON ===
metrics = {
    "version": version,
    "rmse": round(rmse, 2),
    "precision": round(precision, 2),
    "recall": round(recall, 2),
    "threshold": round(threshold, 2)
}

with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# === 9. Uppdatera CHANGELOG ===
with open("CHANGELOG.md", "a", encoding="utf-8") as f:
    f.write("\n## v0.2\n")
    f.write("- Improved model: Ridge Regression (alpha=1.0)\n")
    f.write(f"- RMSE: {rmse:.2f}\n")
    f.write(f"- Precision (high-risk): {precision:.2f}\n")
    f.write(f"- Recall (high-risk): {recall:.2f}\n")
