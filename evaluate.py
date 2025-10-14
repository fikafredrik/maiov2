import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Ladda modellen
model_data = joblib.load("app/model.joblib")
model = model_data["model"]
scaler = model_data["scaler"]

# Ladda datasetet och dela upp
data = load_diabetes(as_frame=True)
X = data.frame.drop(columns=["target"])
y = data.frame["target"]

# Samma split som i träning
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Skala testdata
X_test_scaled = scaler.transform(X_test)

# Prediktion och RMSE
y_pred = model.predict(X_test_scaled)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")
