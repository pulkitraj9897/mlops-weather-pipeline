import pandas as pd
from datetime import datetime

def process_weather():
    # Load raw data
    df = pd.read_csv("data/raw/weather.csv")
    
    # Convert time column to datetime
    df["time"] = pd.to_datetime(df["time"])
    
    # Feature engineering: Extract day/month/hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["hour"] = df["time"].dt.hour
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Save processed data
    df.to_csv("data/processed/weather_clean.csv", index=False)
    print(f"Processed {len(df)} records.")

if __name__ == "__main__":
    process_weather()