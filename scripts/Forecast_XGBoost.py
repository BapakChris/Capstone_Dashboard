import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

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
print(f"[🔁] Running XGBoost forecast with OTB pickup uplift: {timestamp}")

# === LOAD HISTORICAL DATA ===
hist_files = pd.concat([
    pd.read_excel(os.path.join(hist_path, file)).assign(source=file)
    for file in os.listdir(hist_path) if file.endswith(".xlsx")
])

df_hist = hist_files.rename(columns={
    'Date': 'stay_date',
    'Paying Room (Room Sold) Today Actual': 'rooms_occupied',
    'RNA Today': 'available_rooms'
})

df_hist['stay_date'] = pd.to_datetime(df_hist['stay_date'])
df_hist['occupancy'] = df_hist['rooms_occupied'] / df_hist['available_rooms']
df_hist['dow'] = df_hist['stay_date'].dt.dayofweek
df_hist['month'] = df_hist['stay_date'].dt.month
df_hist['is_weekend'] = df_hist['stay_date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int)

# Encode DOW
le = LabelEncoder()
df_hist['dow_encoded'] = le.fit_transform(df_hist['stay_date'].dt.day_name())

# === TRAIN XGBOOST MODEL ===
features = ['month', 'is_weekend', 'dow_encoded']
X = df_hist[features]
y = df_hist['occupancy']
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4)
model.fit(X, y)

# === FORECAST BASE DATES ===
future_dates = pd.date_range(today, periods=90)
df_forecast = pd.DataFrame({'stay_date': future_dates})
df_forecast['dow'] = df_forecast['stay_date'].dt.dayofweek
df_forecast['month'] = df_forecast['stay_date'].dt.month
df_forecast['is_weekend'] = df_forecast['stay_date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int)
df_forecast['dow_encoded'] = le.transform(df_forecast['stay_date'].dt.day_name())

# === BASE XGBOOST FORECAST ===
X_future = df_forecast[features]
base_pred = model.predict(X_future)
df_forecast['base_pred'] = base_pred

# === LOAD TODAY & YESTERDAY OTB ===
otb_today_path = os.path.join(otb_path, f"otb_{today_str}.xlsx")
otb_yesterday_path = os.path.join(otb_path, f"otb_{yesterday_str}.xlsx")

df_otb_today = pd.read_excel(otb_today_path)
df_otb_yesterday = pd.read_excel(otb_yesterday_path)

def clean_otb(df):
    df = df.rename(columns={'Date': 'stay_date', 'Paying Room': 'otb_rooms'})
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

# === MERGE PICKUP INTO FORECAST ===
df_forecast = df_forecast.merge(pickup_df, on='stay_date', how='left')
df_forecast['pickup'] = df_forecast['pickup'].fillna(0)

# === APPLY UPLIFT BASED ON REAL PICKUP ===
df_forecast['forecasted'] = np.clip((df_forecast['base_pred'] * 100 + df_forecast['pickup']), 0, 100).round(2)

# === LOAD ACTUALS (for past stay_dates)
otb_files = sorted([f for f in os.listdir(otb_path) if f.endswith(".xlsx")])
latest_otb = os.path.join(otb_path, otb_files[-1])
df_actuals = pd.read_excel(latest_otb)
df_actuals = df_actuals.rename(columns={
    'Date': 'stay_date',
    'Paying Room': 'otb_rooms',
    'Rooms Available': 'available_rooms'
})
df_actuals['stay_date'] = pd.to_datetime(df_actuals['stay_date'])
df_actuals['occupancy'] = (df_actuals['otb_rooms'] / df_actuals['available_rooms']).round(4)

# === COMBINE ACTUALS + FORECAST ===
df_all = pd.concat([df_actuals[['stay_date', 'occupancy']], df_forecast[['stay_date', 'forecasted']]], ignore_index=True)
df_all = df_all.drop_duplicates('stay_date', keep='last')
df_all['occupancy_forecast'] = np.where(
    df_all['stay_date'] < today,
    df_all['occupancy'] * 100,
    df_all['forecasted']
)

df_all['stay_date'] = df_all['stay_date'].dt.date
df_final = df_all[['stay_date', 'occupancy_forecast']].sort_values('stay_date')

# === EXPORT ===
filename = f"forecast_xgb_{timestamp}.xlsx"
df_final.to_excel(os.path.join(output_path, filename), index=False)
print(f"✅ Final XGBoost forecast (with OTB pickup uplift) saved: {filename}")