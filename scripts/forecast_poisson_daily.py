import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# === SETUP ===
base_path = os.path.expanduser("~/Documents/2025_CAPSTONE")
hist_path = os.path.join(base_path, "data/historical")
otb_path = os.path.join(base_path, "data/otb")
output_path = os.path.join(base_path, "output")
os.makedirs(output_path, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
today = pd.Timestamp.today().normalize()
today_str = today.strftime('%Y%m%d')
yesterday_str = (today - timedelta(days=1)).strftime('%Y%m%d')
print(f"[🔁] Running Poisson forecast with OTB-to-OTB pickup uplift: {timestamp}")

# === LOAD HISTORICAL DATA ===
hist_files = pd.concat([
    pd.read_excel(os.path.join(hist_path, f)).assign(source=f)
    for f in os.listdir(hist_path) if f.endswith(".xlsx")
])

df_hist = hist_files.rename(columns={
    'Date': 'stay_date',
    'Paying Room (Room Sold) Today Actual': 'rooms_occupied',
    'RNA Today': 'available_rooms'
})

df_hist['stay_date'] = pd.to_datetime(df_hist['stay_date'])
df_hist = df_hist.dropna(subset=['stay_date'])
df_hist['month'] = df_hist['stay_date'].dt.month
df_hist['day'] = df_hist['stay_date'].dt.day
df_hist['dow'] = df_hist['stay_date'].dt.dayofweek
df_hist['lag_7d_avg_occupied'] = df_hist['rooms_occupied'].rolling(window=7).mean().shift(1)

# Add lag_1d
lag_df = df_hist[['stay_date', 'rooms_occupied']].copy()
lag_df['stay_date'] = lag_df['stay_date'] + timedelta(days=1)
lag_df = lag_df.rename(columns={'rooms_occupied': 'lag_1d_occupied'})
df_hist = df_hist.merge(lag_df, on='stay_date', how='left')

# === TRAIN POISSON MODEL ===
required_cols = ['rooms_occupied', 'dow', 'month', 'day', 'lag_1d_occupied', 'lag_7d_avg_occupied']
df_train = df_hist.dropna(subset=required_cols)
print(f"📦 Poisson training data rows: {len(df_train)}")

formula = 'rooms_occupied ~ dow + month + day + lag_1d_occupied + lag_7d_avg_occupied'
poisson_model = smf.glm(formula=formula, data=df_train, family=sm.families.Poisson()).fit()

# === LOAD TODAY & YESTERDAY OTB ===
otb_files = sorted([f for f in os.listdir(otb_path) if f.endswith(".xlsx")])
latest_otb = os.path.join(otb_path, f"otb_{today_str}.xlsx")
prev_otb = os.path.join(otb_path, f"otb_{yesterday_str}.xlsx")

df_otb_today = pd.read_excel(latest_otb)
df_otb_yesterday = pd.read_excel(prev_otb)

def clean_otb(df):
    df = df.rename(columns={'Date': 'stay_date', 'Paying Room': 'otb_rooms', 'Rooms Available': 'available_rooms'})
    df['stay_date'] = pd.to_datetime(df['stay_date'], errors='coerce')
    return df.dropna(subset=['stay_date'])

df_otb_today = clean_otb(df_otb_today)
df_otb_yesterday = clean_otb(df_otb_yesterday)

# === COMPUTE OTB PICKUP ===
pickup_df = df_otb_today[['stay_date', 'otb_rooms']].merge(
    df_otb_yesterday[['stay_date', 'otb_rooms']],
    on='stay_date', suffixes=('_today', '_yesterday'), how='left'
)

pickup_df['pickup'] = (pickup_df['otb_rooms_today'] - pickup_df['otb_rooms_yesterday']).clip(lower=0)
pickup_df = pickup_df[['stay_date', 'pickup']]

# === PREPARE TODAY'S OTB FOR FORECAST ===
df_otb = df_otb_today.copy()
df_otb['month'] = df_otb['stay_date'].dt.month
df_otb['day'] = df_otb['stay_date'].dt.day
df_otb['dow'] = df_otb['stay_date'].dt.dayofweek

# Add lag features
lag_lookup = df_hist[['stay_date', 'rooms_occupied']].copy()
lag_lookup['stay_date'] = lag_lookup['stay_date'] + timedelta(days=1)
lag_lookup = lag_lookup.rename(columns={'rooms_occupied': 'lag_1d_occupied'})
df_otb = df_otb.merge(lag_lookup, on='stay_date', how='left').fillna(0)

df_otb = df_otb.merge(df_hist[['stay_date', 'lag_7d_avg_occupied']], on='stay_date', how='left')
df_otb['lag_7d_avg_occupied'] = df_otb['lag_7d_avg_occupied'].fillna(df_hist['lag_7d_avg_occupied'].mean())

# Merge OTB-based pickup
df_otb = df_otb.merge(pickup_df, on='stay_date', how='left')
df_otb['pickup'] = df_otb['pickup'].fillna(0)

# === FORECAST + UPLIFT ===
forecast_rooms = poisson_model.predict(df_otb)
forecast_pct = (forecast_rooms / df_otb['available_rooms']) * 100
df_otb['forecast_poisson'] = (forecast_pct + df_otb['pickup']).round(2)

# Final export
df_otb['occupancy_forecast'] = np.where(
    df_otb['stay_date'] < today,
    (df_otb['otb_rooms'] / df_otb['available_rooms']) * 100,
    df_otb['forecast_poisson']
)

df_export = df_otb[['stay_date', 'occupancy_forecast']].copy()
df_export['stay_date'] = df_export['stay_date'].dt.date

# === EXPORT
output_file = os.path.join(output_path, f"forecast_poisson_daily_{timestamp}.xlsx")
df_export.to_excel(output_file, index=False)
print(f"✅ Final Poisson forecast (with OTB-to-OTB pickup uplift) saved: {output_file}")
