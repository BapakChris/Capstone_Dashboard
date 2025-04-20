# forecast_all_models_daily.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm
import statsmodels.formula.api as smf

# === PATH SETUP ===
base_path = os.path.expanduser("~/Documents/2025_CAPSTONE")
hist_path = os.path.join(base_path, "data/historical")
otb_path = os.path.join(base_path, "data/otb")
output_path = os.path.join(base_path, "output")
os.makedirs(output_path, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
print(f"[🔁] Running combined forecast: {timestamp}")

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

df_hist['stay_date'] = pd.to_datetime(df_hist['stay_date'], errors='coerce')
df_hist = df_hist.dropna(subset=['stay_date'])
df_hist['occupancy'] = (df_hist['rooms_occupied'] / df_hist['available_rooms']) * 100
df_hist['day_of_week'] = df_hist['stay_date'].dt.day_name()
df_hist['month'] = df_hist['stay_date'].dt.month
df_hist['day'] = df_hist['stay_date'].dt.day
df_hist['dow'] = df_hist['stay_date'].dt.dayofweek

# Add lag_1d_occupied
lag_df = df_hist[['stay_date', 'rooms_occupied']].copy()
lag_df['stay_date'] = lag_df['stay_date'] + timedelta(days=1)
lag_df = lag_df.rename(columns={'rooms_occupied': 'lag_1d_occupied'})
df_hist = df_hist.merge(lag_df, on='stay_date', how='left')

# Encode day_of_week for XGBoost
le = LabelEncoder()
df_hist['day_of_week_encoded'] = le.fit_transform(df_hist['day_of_week'])

# === TRAIN POISSON MODEL ===
poisson_formula = 'rooms_occupied ~ dow + month + day + lag_1d_occupied'
required_cols = ['rooms_occupied', 'dow', 'month', 'day', 'lag_1d_occupied']
df_poisson_train = df_hist.dropna(subset=required_cols)
print(f"📦 Poisson training data rows: {len(df_poisson_train)}")
poisson_model = smf.glm(formula=poisson_formula, data=df_poisson_train, family=sm.families.Poisson()).fit()

# === TRAIN XGBOOST MODEL ===
xgb_features = ['month', 'day', 'dow', 'day_of_week_encoded']
X_xgb = df_hist[xgb_features]
y_xgb = df_hist['occupancy']
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4)
xgb_model.fit(X_xgb, y_xgb)

# === LOAD LATEST OTB ===
otb_files = sorted([f for f in os.listdir(otb_path) if f.endswith(".xlsx")])
latest_otb_file = os.path.join(otb_path, otb_files[-1])
df_otb = pd.read_excel(latest_otb_file, engine='openpyxl')
df_otb = df_otb.rename(columns={
    'Date': 'stay_date',
    'Paying Room': 'otb_rooms',  # ✅ Rooms occupied last night (actuals)
    'Rooms Available': 'available_rooms'
})

# 🛠 Clean malformed stay_date like '2025-04-01 Tue'
df_otb['stay_date'] = df_otb['stay_date'].astype(str).str.extract(r'(^\d{4}-\d{2}-\d{2})')[0]
df_otb['stay_date'] = pd.to_datetime(df_otb['stay_date'], errors='coerce')

# Drop any remaining invalid rows
df_otb = df_otb.dropna(subset=['stay_date'])

# Add calendar and model features
df_otb['dow'] = df_otb['stay_date'].dt.dayofweek
df_otb['month'] = df_otb['stay_date'].dt.month
df_otb['day'] = df_otb['stay_date'].dt.day
df_otb['day_of_week'] = df_otb['stay_date'].dt.day_name()
df_otb = df_otb.dropna(subset=['day_of_week'])  # avoid LabelEncoder error
df_otb['day_of_week_encoded'] = le.transform(df_otb['day_of_week'])

# Add lag_1d_occupied to OTB
lag_lookup = df_hist[['stay_date', 'rooms_occupied']].copy()
lag_lookup['stay_date'] = lag_lookup['stay_date'] + timedelta(days=1)
lag_lookup = lag_lookup.rename(columns={'rooms_occupied': 'lag_1d_occupied'})
df_otb = df_otb.merge(lag_lookup, on='stay_date', how='left')

# === MAKE PREDICTIONS ===
df_otb['lag_1d_occupied'] = df_otb['lag_1d_occupied'].fillna(0)
df_otb['forecast_poisson'] = poisson_model.predict(df_otb)
X_forecast_xgb = df_otb[xgb_features]
df_otb['forecast_xgboost'] = np.clip(xgb_model.predict(X_forecast_xgb), 0, 100).round(2)

# Compute actual occupancy if stay_date <= today
today = pd.Timestamp.today().normalize()
df_otb['occupancy'] = np.where(
    df_otb['stay_date'] <= today,
    (df_otb['otb_rooms'] / df_otb['available_rooms'] * 100).round(2),
    np.nan
)

# === ENSEMBLE FORECAST ===
df_otb['forecast_ensemble'] = (
    df_otb['forecast_poisson'] * 0.3 + df_otb['forecast_xgboost'] * 0.7
).round(2)

# === EXPORT FINAL COMPARISON REPORT ===
df_report = df_otb[['stay_date', 'otb_rooms', 'available_rooms', 'occupancy', 'forecast_poisson', 'forecast_xgboost', 'forecast_ensemble']].copy()
df_report['stay_date'] = df_report['stay_date'].dt.date
report_file = os.path.join(output_path, f'forecast_poisson_daily_{timestamp}.xlsx')
df_report.to_excel(report_file, index=False)
print(f"✅ Combined forecast report saved to: {report_file}")

# === EXPORT CHART ===
plt.figure(figsize=(14, 6))
plt.plot(df_report['stay_date'], df_report['forecast_poisson'], label='Poisson', linestyle='--')
plt.plot(df_report['stay_date'], df_report['forecast_xgboost'], label='XGBoost', linestyle='-')
plt.plot(df_report['stay_date'], df_report['forecast_ensemble'], label='Ensemble', linestyle='-.')
plt.plot(df_report['stay_date'], df_report['occupancy'], label='Actual', linestyle=':', color='gray')
plt.title("Forecast Comparison: Poisson vs XGBoost vs Ensemble")
plt.xlabel("Stay Date")
plt.ylabel("Occupancy (%)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
chart_file = os.path.join(output_path, f'forecast_poisson_daily_chart_{timestamp}.png')
plt.savefig(chart_file)
plt.close()
print(f"✅ Chart saved to: {chart_file}")

# === ACCURACY SCORING ===
eval_df = df_report.dropna(subset=['occupancy']).copy()
eval_df['error_poisson'] = (eval_df['forecast_poisson'] - eval_df['occupancy']).abs()
eval_df['error_xgboost'] = (eval_df['forecast_xgboost'] - eval_df['occupancy']).abs()
eval_df['error_ensemble'] = (eval_df['forecast_ensemble'] - eval_df['occupancy']).abs()
eval_df['ape_poisson'] = eval_df['error_poisson'] / eval_df['occupancy']
eval_df['ape_xgboost'] = eval_df['error_xgboost'] / eval_df['occupancy']
eval_df['ape_ensemble'] = eval_df['error_ensemble'] / eval_df['occupancy']

mape_poisson = eval_df['ape_poisson'].mean() * 100
mape_xgb = eval_df['ape_xgboost'].mean() * 100
mape_ens = eval_df['ape_ensemble'].mean() * 100

print(f"📊 MAPE Scores:")
print(f"Poisson: {mape_poisson:.2f}%")
print(f"XGBoost: {mape_xgb:.2f}%")
print(f"Ensemble: {mape_ens:.2f}%")

# Save accuracy breakdown
eval_output = os.path.join(output_path, f'forecast_poisson_daily_accuracy_{timestamp}.xlsx')
eval_df.to_excel(eval_output, index=False)
print(f"📁 Accuracy breakdown saved to: {eval_output}")