#==PART 1: LOAD HISTORICAL PRICING AND ACTUAL OCCUPANCY==#
import os
import pandas as pd
import numpy as np
from datetime import datetime

# === PATH SETUP ===
base_path = "/Users/chrislegaspi/Documents/2025_CAPSTONE"
pricing_path = os.path.join(base_path, "output")
otb_path = os.path.join(base_path, "data/otb")
hist_path = os.path.join(base_path, "data/historical")
os.makedirs(pricing_path, exist_ok=True)

# === LOAD PAST PRICING FILES ===
pricing_files = [f for f in os.listdir(pricing_path) if f.startswith("pricing_recommendation_weighted_") and f.endswith(".xlsx")]
pricing_files = sorted(pricing_files)

pricing_dfs = []
for file in pricing_files:
    df = pd.read_excel(os.path.join(pricing_path, file))
    if 'Recommended BAR Level' not in df.columns:
        continue
    df['source_file'] = file
    pricing_dfs.append(df)

# Combine all pricing logs
pricing_all = pd.concat(pricing_dfs, ignore_index=True)

# Keep only the latest file entry per stay_date
pricing_all = pricing_all.sort_values(by=['stay_date', 'source_file'])
pricing_all = pricing_all.drop_duplicates(subset='stay_date', keep='last')

# Filter to only keep stay dates from today (D0) onward
pricing_all = pricing_all[pricing_all['stay_date'] >= pd.Timestamp.today().normalize()]
pricing_all['stay_date'] = pd.to_datetime(pricing_all['stay_date'])
print(f"✅ Loaded {len(pricing_all)} pricing records from {len(pricing_files)} files.")

#==PART 2: FIX MISSING DTA AND LOAD ACTUAL OCCUPANCY==#
from datetime import date, timedelta

# Fix missing days_to_arrival if needed
# Fix missing days_to_arrival
if 'days_to_arrival' not in pricing_all.columns or pricing_all['days_to_arrival'].isna().any():
    pricing_all['days_to_arrival'] = (pricing_all['stay_date'] - pd.Timestamp.today().normalize()).dt.days

# Load latest OTB file to get actual occupancy
otb_files = sorted([f for f in os.listdir(otb_path) if f.startswith("otb_") and f.endswith(".xlsx")])
otb_dfs = []
for file in otb_files:
    df = pd.read_excel(os.path.join(otb_path, file))
    df = df.rename(columns={'Date': 'stay_date', 'Paying Room': 'rooms_occupied', 'Rooms Available': 'available_rooms'})
    df['stay_date'] = pd.to_datetime(df['stay_date'])
    df = df[['stay_date', 'rooms_occupied', 'available_rooms']]
    otb_dfs.append(df)

otb_all = pd.concat(otb_dfs, ignore_index=True)
otb_all = otb_all.dropna(subset=['stay_date'])
otb_all = otb_all.groupby('stay_date').agg({'rooms_occupied': 'max', 'available_rooms': 'max'}).reset_index()
otb_all['occupancy'] = otb_all['rooms_occupied'] / otb_all['available_rooms']

# Merge with pricing log
pricing_all = pricing_all.merge(otb_all[['stay_date', 'occupancy']], on='stay_date', how='left')

# Preview joined data
print(pricing_all[['stay_date', 'ensemble_forecast', 'days_to_arrival', 'Recommended BAR Level', 'occupancy']].head())

#==PART 2.5: LOAD DOW PICKUP AVERAGE==#
pickup_log_path = os.path.join(pricing_path, "same_day_pickup_dow_log.xlsx")
if os.path.exists(pickup_log_path):
    df_pickup_dow = pd.read_excel(pickup_log_path)
    pickup_avg = df_pickup_dow.groupby('day_of_week')['pickup_rooms'].mean().to_dict()
    pricing_all['day_of_week'] = pricing_all['stay_date'].dt.strftime('%A')
    pricing_all['dow_pickup_avg'] = pricing_all['day_of_week'].map(pickup_avg).fillna(0)
else:
    pricing_all['dow_pickup_avg'] = 0

#==PART 3: CLEAN NA FIELDS BEFORE TRAINING==#

# Drop rows with missing critical values for training
# Fill missing ensemble_forecast with 0.75 (neutral midpoint)
pricing_all['ensemble_forecast'] = pricing_all['ensemble_forecast'].fillna(0.75)

# Ensure BAR Level is numeric
if pricing_all['Recommended BAR Level'].dtype == object:
    pricing_all['Recommended BAR Level'] = pd.to_numeric(pricing_all['Recommended BAR Level'], errors='coerce')

# Drop rows only if other key fields are missing
pricing_all = pricing_all.dropna(subset=['days_to_arrival', 'Recommended BAR Level', 'occupancy'])

print(f"✅ Cleaned data ready for training: {len(pricing_all)} rows")

#==PART 4: MODEL TRAINING: RANDOM FOREST==#
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add DOW as a feature
pricing_all['dow'] = pricing_all['stay_date'].dt.dayofweek

# Define label mapping for BAR levels
distinct_bar_levels = sorted(pricing_all['Recommended BAR Level'].dropna().unique())
bar_mapping = {level: idx for idx, level in enumerate(distinct_bar_levels)}
inv_bar_mapping = {idx: level for level, idx in bar_mapping.items()}
pricing_all['BAR_Label'] = pricing_all['Recommended BAR Level'].map(bar_mapping)

# Define features and target
features = ['ensemble_forecast', 'days_to_arrival', 'dow', 'dow_pickup_avg', 'occupancy']
target = 'BAR_Label'

X = pricing_all[features]
y = pricing_all[target].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
print("📊 Random Forest Classification Report:")
print(classification_report(y_test, y_pred))
print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(pricing_all[['stay_date', 'ensemble_forecast', 'days_to_arrival', 'Recommended BAR Level']].head())

#==PART 5: MAKE LIVE PREDICTIONS==#

# Predict BAR Level using Random Forest
future_df_rf = pricing_all[['stay_date'] + features].copy()
future_df_rf['Predicted BAR Level (RF)'] = rf_model.predict(future_df_rf[features])

print("🔮 Predicted BAR Levels (Random Forest):")
print(future_df_rf[['stay_date', 'ensemble_forecast', 'days_to_arrival', 'Predicted BAR Level (RF)']].head())

#==PART 6: MODEL TRAINING: XGBOOST==#
import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=10, eval_metric='mlogloss', use_label_encoder=False)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
print("📊 XGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))
print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))
# Predict BAR Level using XGBoost
future_df_xgb = pricing_all[['stay_date'] + features].copy()
xgb_preds = xgb_model.predict(future_df_xgb[features])
future_df_xgb['Predicted BAR Level (XGB)'] = [inv_bar_mapping[p] for p in xgb_preds]

print("🔮 Predicted BAR Levels (XGBoost):")

#==PART 7: EXPORT TO EXCEL==#
from datetime import datetime
output_file = os.path.join(pricing_path, f"bar_level_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
# Format stay_date to date-only
final_output = future_df_rf[['stay_date', 'ensemble_forecast', 'days_to_arrival']].copy()
final_output['stay_date'] = pd.to_datetime(final_output['stay_date']).dt.date
final_output['Recommended BAR Level (Weighted)'] = pricing_all['Recommended BAR Level'].values
final_output['Predicted BAR Level (RF)'] = future_df_rf['Predicted BAR Level (RF)']
final_output['Predicted BAR Level (XGB)'] = future_df_xgb['Predicted BAR Level (XGB)']
# Format columns for presentation
final_output = final_output.rename(columns={
    'stay_date': 'Stay Date',
    'ensemble_forecast': 'Ensemble Forecast',
    'days_to_arrival': 'Days to Arrival',
    'Recommended BAR Level (Weighted)': 'Recommended BAR Level (Weighted)',
    'Predicted BAR Level (RF)': 'Predicted BAR Level (RF)',
    'Predicted BAR Level (XGB)': 'Predicted BAR Level (XGB)'
})
final_output.to_excel(output_file, index=False)
print(f"✅ BAR level predictions exported to: {output_file}")
print(future_df_xgb[['stay_date', 'ensemble_forecast', 'days_to_arrival', 'Predicted BAR Level (XGB)']].head())


