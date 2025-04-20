import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# === CONFIG ===
base_path = os.path.expanduser("~/Documents/2025_CAPSTONE")
otb_path = os.path.join(base_path, "data/otb")
forecast_path = os.path.join(base_path, "output")
output_path = os.path.join(base_path, "output")
os.makedirs(output_path, exist_ok=True)

# === DATE SETUP ===
today = pd.Timestamp.today().normalize()
yesterday = today - timedelta(days=1)
today_str = today.strftime("%Y%m%d")
yesterday_str = yesterday.strftime("%Y%m%d")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
target_stay_date = yesterday

# === LOAD ACTUAL OCCUPANCY FROM TODAY'S OTB ===
otb_file = os.path.join(otb_path, f"otb_{today_str}.xlsx")
df_otb = pd.read_excel(otb_file)
df_otb = df_otb.rename(columns={'Date': 'stay_date', 'Paying Room': 'otb_rooms', 'Rooms Available': 'available_rooms'})
df_otb['stay_date'] = pd.to_datetime(df_otb['stay_date'])
df_otb['occupancy'] = (df_otb['otb_rooms'] / df_otb['available_rooms'] * 100).round(2)
df_actual = df_otb[df_otb['stay_date'] == target_stay_date].copy()

# === HELPER: FIND FORECAST FILE FOR A MODEL
def find_forecast_value(prefix, stay_date):
    try:
        files = [f for f in os.listdir(forecast_path) if f.startswith(f"{prefix}_{yesterday_str}") and f.endswith(".xlsx")]
        if not files:
            print(f"⚠️ No file found for {prefix} on {yesterday_str}")
            return np.nan
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(forecast_path, x)))
        df = pd.read_excel(os.path.join(forecast_path, latest_file))
        df['stay_date'] = pd.to_datetime(df['stay_date'])
        row = df[df['stay_date'] == stay_date]
        return float(row.iloc[0]['occupancy_forecast']) if not row.empty else np.nan
    except Exception as e:
        print(f"❌ Error reading {prefix}: {e}")
        return np.nan

# === LOAD FORECAST VALUES
poisson_val = find_forecast_value("forecast_poisson_daily", target_stay_date)
xgb_val     = find_forecast_value("forecast_xgb", target_stay_date)
knn_val     = find_forecast_value("forecast_knn", target_stay_date)

# === BUILD FINAL ACCURACY REPORT
df_accuracy = pd.DataFrame({
    "stay_date": [target_stay_date.date()],
    "occupancy": df_actual["occupancy"].values if not df_actual.empty else [np.nan],
    "poisson": [poisson_val],
    "xgb": [xgb_val],
    "knn": [knn_val]
})

df_accuracy["ensemble"] = df_accuracy[["poisson", "xgb", "knn"]].mean(axis=1).round(2)
df_accuracy["variance"] = (df_accuracy["ensemble"] - df_accuracy["occupancy"]).round(2)
df_accuracy["ape"] = (np.abs(df_accuracy["variance"]) / df_accuracy["occupancy"]).round(4)
df_accuracy["mape"] = (df_accuracy["ape"] * 100).round(2)

# === EXPORT REPORT
output_file = os.path.join(output_path, f"forecast_soft_ensemble_accuracy_{timestamp}.xlsx")
df_accuracy.to_excel(output_file, index=False)
print(f"✅ Accuracy report saved: {output_file}")

# === APPEND TO LONG-TERM LOG
log_path = os.path.join(output_path, "mape_log_soft_ensemble.xlsx")
try:
    if os.path.exists(log_path):
        df_log = pd.read_excel(log_path)
        if 'stay_date' in df_log.columns:
            df_log['stay_date'] = pd.to_datetime(df_log['stay_date']).dt.date
            df_accuracy['stay_date'] = pd.to_datetime(df_accuracy['stay_date']).dt.date
            df_combined = pd.concat([df_log, df_accuracy]).drop_duplicates('stay_date').sort_values('stay_date')
        else:
            print("⚠️ Log file found but missing 'stay_date' column — overwriting.")
            df_combined = df_accuracy.copy()
    else:
        df_combined = df_accuracy.copy()

    df_combined.to_excel(log_path, index=False)
    print(f"📈 Log updated: {log_path}")
except Exception as e:
    print(f"❌ Failed to update log file: {e}")



