# scripts/fetch_weather.py
import requests
import pandas as pd
from datetime import datetime

def fetch_weather(latitude=28.61, longitude=77.23, days=30):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&past_days={days}&hourly=temperature_2m,relativehumidity_2m,precipitation"
    response = requests.get(url).json()
    data = pd.DataFrame(response["hourly"])
    data["fetch_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data.to_csv("data/raw/weather.csv", index=False)
    print(f"Saved {len(data)} records.")

if __name__ == "__main__":
    fetch_weather()