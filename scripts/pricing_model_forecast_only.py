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

# === LOAD SAME-DAY PICKUP LOG ===
pickup_log_path = os.path.join(output_path, "same_day_pickup_dow_log.xlsx")
df_pickup_log = pd.read_excel(pickup_log_path) if os.path.exists(pickup_log_path) else pd.DataFrame(columns=['stay_date', 'day_of_week', 'pickup_rooms'])
df_pickup_log['stay_date'] = pd.to_datetime(df_pickup_log['stay_date'], errors='coerce').dt.date

# === FIND AND LOAD LATEST ENSEMBLE FORECAST FILE FOR TODAY ===
pattern = os.path.join(output_path, f"forecast_soft_ensemble_quality_{today_str}_*.xlsx")
forecast_files = glob.glob(pattern)
if not forecast_files:
    raise FileNotFoundError(f"No forecast file found for today: {today_str}")
latest_forecast_file = max(forecast_files, key=os.path.getctime)
print(f"\U0001F4C2 Using forecast file: {latest_forecast_file}")
df_forecast = pd.read_excel(latest_forecast_file)
df_forecast['stay_date'] = pd.to_datetime(df_forecast['stay_date']).dt.date
df_forecast['ensemble_forecast'] = df_forecast['ensemble_forecast'] / 100

#==PART 2==#
df_matrix = pd.read_excel(os.path.join(data_path, "ratematrix", "rm_neopuriindah.xlsx"))

#==PART 3==#
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

#==PART 4==#
rd_file = os.path.join(data_path, "Rate Deployment Report", f"rd_{today_str}.xlsx")
df_rd = pd.read_excel(rd_file)
df_rd['Date'] = pd.to_datetime(df_rd['Date'], dayfirst=True).dt.date
df_rd = df_rd.rename(columns={'Room Type': 'Room_Type', 'Rate (Gross)': 'CM Rate'})
priority_order = ['Superior Room', 'Deluxe City View', 'Junior Suite']
df_rd = df_rd[df_rd['Room_Type'].isin(priority_order) & (df_rd['Availability'] > 0)]
df_rd['Room_Type'] = pd.Categorical(df_rd['Room_Type'], categories=priority_order, ordered=True)
df_rd = df_rd.sort_values(['Date', 'Room_Type'])
df_rd_selected = df_rd.drop_duplicates(subset=['Date'], keep='first')

#==PART 5==#
holiday_file = os.path.join(data_path, "Others", "jakarta_holidays_2025.xlsx")
df_holidays = pd.read_excel(holiday_file)
df_holidays['Date'] = pd.to_datetime(df_holidays['Date'], dayfirst=True).dt.date
holiday_dates = df_holidays['Date'].tolist()

#==PART 6==#
df_all = df_forecast.merge(df_rs[['Date', 'Deployed Rate', 'weighted_compset']], left_on='stay_date', right_on='Date', how='left')
df_all = df_all.merge(df_rd_selected[['Date', 'Room_Type', 'CM Rate']], left_on='stay_date', right_on='Date', how='left')
df_all.drop(columns=['Date_x', 'Date_y'], inplace=True)

# === Merge sales and pickup data ===
df_base = df_rd[df_rd['Room_Type'] == 'Superior Room']
df_base_merge = df_base[['Date', 'Availability', '% Rooms Sold']].rename(columns={'Date': 'stay_date', 'Availability': 'Unsold Rooms', '% Rooms Sold': 'Base Rm % Sold'})
df_all = df_all.merge(df_base_merge, on='stay_date', how='left')
df_all = df_all.merge(df_pickup_log[['stay_date', 'pickup_rooms']], on='stay_date', how='left')

cutoff = today + timedelta(days=30)
df_all = df_all[(df_all['stay_date'] >= today) & (df_all['stay_date'] <= cutoff)]
df_all['Holiday'] = np.where(df_all['stay_date'].isin(holiday_dates), 'Yes', 'No')

# === Score logic ===
df_all['Base Rm % Sold'] = df_all['Base Rm % Sold'].fillna(0)
df_all['ensemble_forecast'] = df_all['ensemble_forecast'].fillna(0)
df_all['blended_score'] = (df_all['Base Rm % Sold'] / 100 + df_all['ensemble_forecast']) / 2

# === Forecast vs Pickup Delta ===
df_all['Pickup Delta (%)'] = (df_all['Base Rm % Sold'] - (df_all['ensemble_forecast'] * 100)).round(2)
def classify_pacing(delta):
    if delta > 5:
        return 'Ahead of Forecast'
    elif delta < -5:
        return 'Behind Forecast'
    else:
        return 'On Track'
df_all['Pacing Status'] = df_all['Pickup Delta (%)'].apply(classify_pacing)

# === Adjust blended score by same-day pickup pace ===
def adjust_blended_with_pickup_gap(row):
    if row['stay_date'] != today:
        return row['blended_score']
    unsold = row.get('Unsold Rooms', 0)
    pickup_trend = row.get('pickup_rooms', 0)
    pickup_gap = unsold - pickup_trend
    if pickup_gap <= -10:
        return min(row['blended_score'] + 0.05, 1.0)
    elif pickup_gap >= 10:
        return max(row['blended_score'] - 0.03, 0.0)
    else:
        return row['blended_score']
df_all['blended_score'] = df_all.apply(adjust_blended_with_pickup_gap, axis=1)

# === Uplift and Pricing ===
def uplift_multiplier(blended):
    if blended <= 0.50:
        return 1.03
    elif blended <= 0.60:
        return 1.05
    elif blended <= 0.70:
        return 1.07
    elif blended <= 0.90:
        return 1.08
    else:
        return 1.10
df_all['uplift_multiplier'] = df_all['blended_score'].apply(uplift_multiplier)
df_all['Recommended Rate'] = (df_all['weighted_compset'] * df_all['uplift_multiplier']).round(0)
df_all['Recommended Rate Override'] = np.nan

def final_rate(row):
    try:
        override = float(row['Recommended Rate Override']) if not pd.isna(row['Recommended Rate Override']) else None
    except:
        override = None
    base_rate = override if override is not None else row['Recommended Rate']
    floor = 0
    bar_8_row = df_matrix[df_matrix['BAR Level'] == 8]
    room_col = f"RO_{row['Room_Type']}"
    if not bar_8_row.empty and room_col in bar_8_row.columns:
        bar_8_rate = bar_8_row[room_col].values[0]
        floor = bar_8_rate * 0.85
    return max(base_rate, floor)
df_all['Final Rate'] = df_all.apply(final_rate, axis=1).round(0)

# === ADD RATE SENSITIVITY FLAG ===
def rate_sensitivity(row):
    try:
        if row['uplift_multiplier'] >= 1.08 and row['Base Rm % Sold'] < 30:
            return 'High Risk (Check)'
        elif row['uplift_multiplier'] <= 1.05 and row['Base Rm % Sold'] >= 70:
            return 'Too Soft?'
        else:
            return 'Normal'
    except:
        return 'Unknown'
df_all['Rate Sensitivity'] = df_all.apply(rate_sensitivity, axis=1)

# === Rate Source Tag ===
def identify_rate_source(row):
    try:
        override = float(row['Recommended Rate Override']) if not pd.isna(row['Recommended Rate Override']) else None
    except:
        override = None
    bar_8_row = df_matrix[df_matrix['BAR Level'] == 8]
    room_col = f"RO_{row['Room_Type']}"
    floor = 0
    if not bar_8_row.empty and room_col in bar_8_row.columns:
        bar_8_rate = bar_8_row[room_col].values[0]
        floor = bar_8_rate * 0.85
    if override:
        return 'Override'
    elif row['Final Rate'] < floor:
        return 'BAR Floor'
    else:
        return 'Uplift'
df_all['Rate Source'] = df_all.apply(identify_rate_source, axis=1)

# === BAR Recommendation ===
def get_recommended_bar(compset_rate, matrix, room_type):
    bar_col = f'RO_{room_type}'
    if bar_col not in matrix.columns:
        return 'BAR ADHOC'
    for i in range(1, 9):
        bar_row = matrix[matrix['BAR Level'] == i]
        if not bar_row.empty and bar_col in bar_row:
            bar_rate = bar_row[bar_col].values[0]
            if compset_rate >= bar_rate:
                return i
    return 'BAR ADHOC'
df_all['Recommended BAR'] = df_all.apply(lambda row: get_recommended_bar(row['weighted_compset'], df_matrix, row['Room_Type']), axis=1)

# === Action Statement ===
def generate_action(row):
    try:
        uplift_pct = int((row['uplift_multiplier'] - 1) * 100)
        return f"You still have {int(row['Unsold Rooms'])} Superior Rooms, Sell it at {uplift_pct}% higher than compset"
    except:
        return "Check data"
df_all['Action'] = df_all.apply(generate_action, axis=1)

# === Recompute columns before export ===
df_all['Discount Applied'] = df_all['CM Rate'] - df_all['Deployed Rate']
df_all['Variance vs Compset'] = df_all['Deployed Rate'] - df_all['weighted_compset']
df_all['Final Rate Raw'] = df_all['Final Rate']  # store raw before formatting

# === BUILD TOP SUMMARY ROW BEFORE FORMATTING ===
summary_data = {
    'stay_date': 'SUMMARY',
    'Room_Type': '',
    'weighted_compset': '',
    'Deployed Rate': '',
    'CM Rate': '',
    'Discount Applied': '',
    'Variance vs Compset': '',
    'Recommended BAR': '',
    'uplift_multiplier': '',
    'Recommended Rate': '',
    'Recommended Rate Override': '',
    'Final Rate': f"{int(pd.to_numeric(df_all[df_all['stay_date'] != 'SUMMARY']['Final Rate Raw'], errors='coerce').mean()):,}",
    'Rate Source': '',
    'Pickup Delta (%)': '',
    'Pacing Status': f"{(df_all['Pacing Status'] == 'Behind Forecast').sum()} behind",
    'Holiday': '',
    'Base Rm % Sold': round(df_all['Base Rm % Sold'].mean(), 2),
    'Action': f"Avg Forecast: {round(df_all['ensemble_forecast'].mean() * 100, 2)}% | Rooms Left Today: {df_all[df_all['stay_date'] == today]['Unsold Rooms'].sum()}"
}
df_all = pd.concat([pd.DataFrame([summary_data]), df_all], ignore_index=True)

# === FORMAT NUMERIC COLUMNS SAFELY, SKIP SUMMARY ROW ===
for col in ['weighted_compset', 'Deployed Rate', 'CM Rate',
            'Discount Applied', 'Variance vs Compset', 'Recommended Rate', 'Final Rate']:
    df_all.loc[df_all['stay_date'] != 'SUMMARY', col] = (
        df_all.loc[df_all['stay_date'] != 'SUMMARY', col]
        .fillna(0)
        .round(0)
        .astype(int)
        .map('{:,}'.format)
    )

# === EXPORT ===
final_cols = [
    'stay_date', 'Room_Type', 'weighted_compset', 'Deployed Rate', 'CM Rate',
    'Discount Applied', 'Variance vs Compset', 'Recommended BAR',
    'uplift_multiplier', 'Recommended Rate', 'Recommended Rate Override',
    'Final Rate', 'Final Rate Raw', 'Rate Source', 'Pickup Delta (%)', 'Pacing Status',
    'Rate Sensitivity', 'Holiday', 'Base Rm % Sold', 'Action'
]

output_file = os.path.join(output_path, f"pricing_recommendation_{timestamp}.xlsx")
df_all.to_excel(output_file, index=False, columns=final_cols)
print(f"\u2705 Pricing recommendation saved to: {output_file}")
