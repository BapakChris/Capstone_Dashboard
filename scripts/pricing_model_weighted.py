#==PART 1==#
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# === PATH SETUP ===
base_path = "/Users/chrislegaspi/Documents/2025_CAPSTONE"
output_path = os.path.join(base_path, "output")
data_path = os.path.join(base_path, "data")
today = datetime.today().date()
today_str = today.strftime("%Y%m%d")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

total_rooms = 96

# === LOAD FORECAST AND PICKUP ===
pickup_log_path = os.path.join(output_path, "same_day_pickup_dow_log.xlsx")
df_pickup_log = pd.read_excel(pickup_log_path) if os.path.exists(pickup_log_path) else pd.DataFrame(columns=['stay_date', 'day_of_week', 'pickup_rooms'])
df_pickup_log['stay_date'] = pd.to_datetime(df_pickup_log['stay_date'], errors='coerce').dt.date

pattern = os.path.join(output_path, f"forecast_soft_ensemble_quality_{today_str}_*.xlsx")
forecast_files = glob.glob(pattern)
if not forecast_files:
    raise FileNotFoundError(f"No forecast file found for today: {today_str}")
latest_forecast_file = max(forecast_files, key=os.path.getctime)
print(f"\U0001F4C2 Using forecast file: {latest_forecast_file}")
df_forecast = pd.read_excel(latest_forecast_file)
# Filter for stay dates from today (D0) onward

df_forecast['stay_date'] = pd.to_datetime(df_forecast['stay_date']).dt.date

# === LOAD COMPSET RATES ===
rs_file = os.path.join(data_path, "rateshopping", f"rs_{today_str}.xlsx")
df_rs = pd.read_excel(rs_file)
df_rs.columns = df_rs.columns.str.strip()
df_rs['Date'] = pd.to_datetime(df_rs['Date'], dayfirst=True).dt.date
df_rs = df_rs.rename(columns={'Hotel Neo Puri Indah - Jakarta': 'Deployed Rate'})
df_rs['weighted_compset'] = sum(df_rs[c] * w for c, w in {
    'Brits Hotel Puri Indah': 0.50,
    'Maple Hotel Grogol': 0.30,
    'All Nite & Day Kebon Jeruk': 0.20
}.items())
df_rs = df_rs[['Date', 'weighted_compset']]
df_forecast = df_forecast.merge(df_rs, left_on='stay_date', right_on='Date', how='left')
df_forecast['weighted_compset'] = df_forecast['weighted_compset'].apply(lambda x: f"{x:,.0f}" if not pd.isna(x) else '')
df_forecast.drop(columns=['Date'], inplace=True)

# === PROJECTED OCCUPANCY ===
df_forecast = df_forecast[df_forecast['stay_date'] >= today].copy()
df_forecast['ensemble_forecast'] = (df_forecast['ensemble_forecast'] / 100).round(4)
pickup_avg = df_pickup_log.groupby('day_of_week')['pickup_rooms'].mean().to_dict()
df_forecast['day_of_week'] = df_forecast['stay_date'].apply(lambda d: d.strftime('%A'))

df_forecast['projected_occupancy'] = df_forecast['ensemble_forecast']
df_forecast['days_to_arrival'] = (pd.to_datetime(df_forecast['stay_date']) - pd.to_datetime(today)).dt.days

# === LOAD RATE MATRIX ===
df_matrix = pd.read_excel(os.path.join(data_path, "ratematrix", "rm_neopuriindah.xlsx"))

# === RECOMMEND BAR LEVEL ===
def recommend_bar_level(row):
    occ = row['ensemble_forecast']
    dta = row['days_to_arrival']
    dow = row['day_of_week']

    # Smart layered logic
    if dta <= 1 and occ >= 0.90:
        return 1
    elif dta <= 3 and occ >= 0.85:
        return 2
    elif dta <= 7 and occ >= 0.80:
        return 3
    elif dta <= 14 and occ >= 0.75:
        return 4
    elif dta <= 21 and occ >= 0.70:
        return 5
    elif occ >= 0.65:
        return 6
    elif occ >= 0.60:
        return 7
    else:
        return 8

df_forecast['Recommended BAR Level'] = df_forecast.apply(recommend_bar_level, axis=1)

# === EXTRACT BAR RATE ===
def extract_bar_rate(row):
    room_col = 'Superior Room'
    level = row['Recommended BAR Level']
    bar_row = df_matrix[df_matrix['BAR Level'] == f"BAR {level}"]
    if not bar_row.empty and room_col in bar_row.columns:
        return bar_row[room_col].values[0]
    return np.nan

df_forecast['BAR Rate (Net)'] = df_forecast.apply(extract_bar_rate, axis=1)
df_forecast['BAR Rate'] = df_forecast['BAR Rate (Net)'].apply(lambda x: f"{x:,.0f}")
df_forecast['Final Rate'] = df_forecast['BAR Rate']
df_forecast['Rate Source'] = 'Strategic BAR Rule'

# Add rate decision reason tag

def rate_decision_reason(row):
    level = row['Recommended BAR Level']
    dta = row['days_to_arrival']
    occ = row['ensemble_forecast']

    if level == 1:
        return 'Last minute compression (D0–1, Occ ≥ 90%)'
    elif level == 2:
        return 'Short window surge (D1–3, Occ ≥ 85%)'
    elif level == 3:
        return 'Strong pace within week (D1–7, Occ ≥ 80%)'
    elif level == 4:
        return 'Healthy D8–14 pace (Occ ≥ 75%)'
    elif level == 5:
        return 'Steady demand ahead (Occ ≥ 70%)'
    elif level == 6:
        return 'Soft zone buffer (Occ ≥ 65%)'
    elif level == 7:
        return 'Hold low-tier rate - conservative move (Occ ≥ 60%)'
    else:
        return 'Distress rate fallback (Occ < 60%)'

df_forecast['Rate Decision Reason'] = df_forecast.apply(rate_decision_reason, axis=1)

# === EXPORT ===
final_cols = [
    'stay_date', 'day_of_week', 'ensemble_forecast', 'days_to_arrival',
    'weighted_compset', 'Recommended BAR Level', 'BAR Rate',
    'Rate Source', 'Rate Decision Reason'
]

output_file = os.path.join(output_path, f"pricing_recommendation_weighted_{today_str}_{timestamp}.xlsx")
df_forecast.to_excel(output_file, index=False, columns=final_cols)
print(f"✅ Strategic BAR pricing saved to: {output_file}")
