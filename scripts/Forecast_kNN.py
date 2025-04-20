import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.neighbors import KNeighborsRegressor

# === PATHS ===
base_path = "/Users/chrislegaspi/Documents/2025_CAPSTONE"
paths = {
    "historical": os.path.join(base_path, "data/historical"),
    "otb": os.path.join(base_path, "data/otb"),
    "output": os.path.join(base_path, "output")
}
os.makedirs(paths["output"], exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
today = pd.Timestamp.today().normalize()
today_str = today.strftime('%Y%m%d')
yesterday_str = (today - timedelta(days=1)).strftime('%Y%m%d')

# === Load historical data ===
def load_historical_data(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".xlsx")]
    df_list = []
    for file in files:
        df = pd.read_excel(os.path.join(folder, file))
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w\s]", "", regex=True)
        col_date = next((c for c in df.columns if c.startswith("date")), None)
        col_occ = next((c for c in df.columns if "room_occupied_today" in c), None)
        col_rna = next((c for c in df.columns if "rna_today" in c), None)
        if col_date and col_occ and col_rna:
            df['stay_date'] = pd.to_datetime(df[col_date])
            df['rooms_occupied'] = df[col_occ]
            df['total_rooms'] = df[col_rna]
            df['occupancy'] = df['rooms_occupied'] / df['total_rooms']
            df_list.append(df[['stay_date', 'occupancy']])
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

# === Load OTB actuals ===
def load_otb_file(filename):
    df = pd.read_excel(os.path.join(paths["otb"], filename))
    df = df.rename(columns={
        'Date': 'stay_date',
        'Paying Room': 'otb_rooms',
        'Rooms Available': 'available_rooms'
    })
    df['stay_date'] = pd.to_datetime(df['stay_date'], errors='coerce')
    df['occupancy'] = (df['otb_rooms'] / df['available_rooms']).round(4)
    return df[['stay_date', 'occupancy', 'otb_rooms']].dropna()

df_hist = load_historical_data(paths["historical"])
df_otb_today = load_otb_file(f"otb_{today_str}.xlsx")
df_otb_yesterday = load_otb_file(f"otb_{yesterday_str}.xlsx")

# === Compute pickup from OTB
pickup_df = df_otb_today.merge(df_otb_yesterday, on='stay_date', suffixes=('_today', '_yesterday'))
pickup_df['pickup'] = (pickup_df['otb_rooms_today'] - pickup_df['otb_rooms_yesterday']).clip(lower=0)
pickup_df = pickup_df[['stay_date', 'pickup']]

# === Train kNN
def train_knn(df_hist, k=5):
    df = df_hist.copy()
    df['dayofyear'] = df['stay_date'].dt.dayofyear
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(df[['dayofyear']], df['occupancy'])
    return model

# === Forecast future with uplift
def forecast_knn_with_uplift(model, pickup_by_stay_date, days_ahead=90):
    future_dates = [today + timedelta(days=i) for i in range(days_ahead)]
    df_forecast = pd.DataFrame({'stay_date': pd.to_datetime(future_dates)})
    df_forecast['dayofyear'] = df_forecast['stay_date'].dt.dayofyear
    base = model.predict(df_forecast[['dayofyear']])
    pickup_uplift = df_forecast['stay_date'].map(pickup_by_stay_date).fillna(0)
    df_forecast['occupancy_forecast'] = np.clip((base * 100 + pickup_uplift), 0, 100).round(2)
    return df_forecast[['stay_date', 'occupancy_forecast']]

if not df_hist.empty and not df_otb_today.empty and not df_otb_yesterday.empty:
    pickup_by_date = dict(zip(pickup_df['stay_date'], pickup_df['pickup']))
    knn_model = train_knn(df_hist)
    df_future = forecast_knn_with_uplift(knn_model, pickup_by_date)

    # Combine actuals (past) + forecast (future)
    df_actuals = df_otb_today[df_otb_today['stay_date'] < today][['stay_date', 'occupancy']]
    df_actuals['occupancy_forecast'] = df_actuals['occupancy'] * 100
    df_actuals = df_actuals[['stay_date', 'occupancy_forecast']]

    df_final = pd.concat([df_actuals, df_future], ignore_index=True)
    df_final['stay_date'] = df_final['stay_date'].dt.date
    df_final = df_final.sort_values('stay_date')

    output_file = f"forecast_knn_{timestamp}.xlsx"
    df_final.to_excel(os.path.join(paths["output"], output_file), index=False)
    print(f"✅ Final k-NN forecast (with actuals + OTB pickup uplift) saved: {output_file}")
else:
    print("❌ Missing data: historical or OTB file not found.")
