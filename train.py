import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import mlflow
from sklearn.model_selection import train_test_split  

# Ensure folders exist
os.makedirs("models", exist_ok=True)

# Load processed data
df = pd.read_csv("data/processed/weather_clean.csv")
X = df[["temperature_2m", "relativehumidity_2m"]]
y = (df["precipitation"] > 0).astype(int)  # 1 if rain, 0 otherwise

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save model
model = LogisticRegression()
model.fit(X_train, y_train)
joblib.dump(model, "models/rain_predictor.pkl")

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#Log with MLflow
with mlflow.start_run():
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    mlflow.sklearn.log_model(model, "model")