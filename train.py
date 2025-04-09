import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Create folders if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load and train
df = pd.read_csv("data/processed/weather_clean.csv")
X = df[["temperature_2m", "relativehumidity_2m"]]
y = (df["precipitation"] > 0).astype(int)

model = LogisticRegression().fit(X, y)
joblib.dump(model, "models/rain_predictor.pkl")
print("Model saved to models/rain_predictor.pkl")