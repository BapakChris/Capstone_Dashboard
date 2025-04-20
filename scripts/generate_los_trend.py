import pandas as pd
from datetime import timedelta
import os

# === CONFIG ===
base_path = "/Users/chrislegaspi/Documents/2025_CAPSTONE"
pickup_path = os.path.join(base_path, "data/pickup")
output_path = os.path.join(base_path, "output")
os.makedirs(output_path, exist_ok=True)

pickup_files = sorted([f for f in os.listdir(pickup_path) if f.startswith("pickup_") and f.endswith(".xlsx")])
all_rows = []

for file in pickup_files:
    df = pd.read_excel(os.path.join(pickup_path, file))
    df = df[['Arrival Date', 'Departure Date', 'Room(s)']].dropna()
    df['Arrival Date'] = pd.to_datetime(df['Arrival Date'])
    df['Departure Date'] = pd.to_datetime(df['Departure Date'])
    df['LOS'] = (df['Departure Date'] - df['Arrival Date']).dt.days
    df = df[df['LOS'] > 0]

    for _, row in df.iterrows():
        for offset in range(row['LOS']):
            stay_date = (row['Arrival Date'] + timedelta(days=offset)).date()  # Convert to date
            all_rows.append({'stay_date': stay_date, 'LOS': row['LOS']})

# === SAVE TO EXCEL ===
df_expanded = pd.DataFrame(all_rows)
los_trend = df_expanded.groupby('stay_date')['LOS'].mean().round(2).reset_index()
los_trend['stay_date'] = pd.to_datetime(los_trend['stay_date']).dt.date  # Ensure no time in output
los_trend.to_excel(os.path.join(output_path, "los_trend.xlsx"), index=False)

print("✅ LOS trend saved with clean date format.")

