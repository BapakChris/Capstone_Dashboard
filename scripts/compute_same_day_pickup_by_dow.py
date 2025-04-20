# compute_same_day_pickup_by_dow.py (REVISED)

import os
import pandas as pd
from datetime import datetime, timedelta
import re

# === SETUP ===
base_path = os.path.expanduser("~/Documents/2025_CAPSTONE")
otb_path = os.path.join(base_path, "data/otb")
output_path = os.path.join(base_path, "output")
output_file = os.path.join(output_path, "same_day_pickup_dow_log.xlsx")
os.makedirs(output_path, exist_ok=True)

# === LOAD OTB FILES ===
otb_files = sorted([f for f in os.listdir(otb_path) if f.startswith("otb_") and f.endswith(".xlsx")])

# Extract unique date part (YYYYMMDD) from filenames with optional time suffixes
def extract_date(filename):
    match = re.search(r"otb_(\d{8})", filename)
    return datetime.strptime(match.group(1), "%Y%m%d").date() if match else None

file_map = {}
for f in otb_files:
    date_part = extract_date(f)
    if date_part:
        file_map.setdefault(date_part, []).append(f)

# Sort and pair dates
sorted_dates = sorted(file_map.keys())
pickup_rows = []

for i in range(1, len(sorted_dates)):
    date_today = sorted_dates[i]
    date_yesterday = sorted_dates[i - 1]

    # Use the latest file for each date
    file_today = max(file_map[date_today], key=lambda x: os.path.getmtime(os.path.join(otb_path, x)))
    file_yesterday = max(file_map[date_yesterday], key=lambda x: os.path.getmtime(os.path.join(otb_path, x)))

    try:
        df_today = pd.read_excel(os.path.join(otb_path, file_today))
        df_yesterday = pd.read_excel(os.path.join(otb_path, file_yesterday))
    except Exception as e:
        print(f"⚠️ Skipping {date_today.strftime('%Y-%m-%d')} due to file error: {e}")
        continue

    df_today = df_today.rename(columns={'Date': 'stay_date', 'Paying Room': 'rooms_today'})
    df_yesterday = df_yesterday.rename(columns={'Date': 'stay_date', 'Paying Room': 'rooms_yesterday'})
    df_today['stay_date'] = pd.to_datetime(df_today['stay_date'], errors='coerce').dt.date
    df_yesterday['stay_date'] = pd.to_datetime(df_yesterday['stay_date'], errors='coerce').dt.date

    stay_date = date_yesterday

    row_today = df_today[df_today['stay_date'] == stay_date]
    row_yesterday = df_yesterday[df_yesterday['stay_date'] == stay_date]

    if row_today.empty or row_yesterday.empty:
        continue

    try:
        pickup = int(row_today.iloc[0]['rooms_today']) - int(row_yesterday.iloc[0]['rooms_yesterday'])
        pickup = max(pickup, 0)
    except Exception as e:
        print(f"⚠️ Error calculating pickup for {stay_date}: {e}")
        continue

    pickup_rows.append({
        'stay_date': stay_date,
        'day_of_week': stay_date.strftime('%A'),
        'pickup_rooms': pickup
    })

# === CREATE AND SAVE FINAL LOG ===
df_pickup = pd.DataFrame(pickup_rows)
df_pickup = df_pickup.drop_duplicates(subset='stay_date').sort_values('stay_date')
df_pickup.to_excel(output_file, index=False)

print(f"✅ Final same-day pickup log saved to: {output_file}")
