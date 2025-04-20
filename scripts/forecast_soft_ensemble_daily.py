import os
import pandas as pd
import numpy as np
from datetime import datetime

# === CONFIG ===
base_path = os.path.expanduser("~/Documents/2025_CAPSTONE")
output_path = os.path.join(base_path, "output")
os.makedirs(output_path, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
today = pd.Timestamp.today().normalize()

# === LOAD LATEST FORECAST FILES ===
def load_latest_forecast(prefix):
    files = [f for f in os.listdir(output_path) if f.startswith(prefix) and f.endswith(".xlsx")]
    if not files:
        raise FileNotFoundError(f"No forecast file found for {prefix}")
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(output_path, f)))
    df = pd.read_excel(os.path.join(output_path, latest_file))
    df['stay_date'] = pd.to_datetime(df['stay_date'])
    df = df.rename(columns={'occupancy_forecast': prefix})
    return df[['stay_date', prefix]]

df_poisson = load_latest_forecast("forecast_poisson_daily")
df_xgb = load_latest_forecast("forecast_xgb")
df_knn = load_latest_forecast("forecast_knn")

# === MERGE FORECASTS ===
df = df_poisson.merge(df_xgb, on='stay_date', how='outer')
df = df.merge(df_knn, on='stay_date', how='outer')

# === WEIGHTED ENSEMBLE FORECAST ===
weights = {
    'forecast_poisson_daily': 0.1,
    'forecast_xgb': 0.6,
    'forecast_knn': 0.3
}

df['ensemble_forecast'] = (
    df['forecast_poisson_daily'] * weights['forecast_poisson_daily'] +
    df['forecast_xgb'] * weights['forecast_xgb'] +
    df['forecast_knn'] * weights['forecast_knn']
).round(2)

# === FINAL OUTPUT ===
df_output = df[['stay_date', 'ensemble_forecast']].copy()
df_output = df_output.sort_values('stay_date')
df_output['stay_date'] = df_output['stay_date'].dt.date

# === SAVE TO EXCEL ===
output_file = os.path.join(output_path, f"forecast_soft_ensemble_quality_{timestamp}.xlsx")
df_output.to_excel(output_file, index=False)
print(f"✅ Final ensemble forecast saved: {output_file}")
