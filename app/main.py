from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/rain_predictor.pkl")

@app.post("/predict")
def predict(temperature: float, humidity: float):
    prediction = model.predict([[temperature, humidity]])[0]
    return {"will_rain": bool(prediction)}