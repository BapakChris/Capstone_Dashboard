import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
import xgboost as xgb
import matplotlib.pyplot as plt

# === PATH SETUP ===
base_path = os.path.expanduser("~/Documents/2025_CAPSTONE")
hist_path = os.path.join(base_path, "data/historical")
output_path = os.path.join(base_path, "output")
os.makedirs(output_path, exist_ok=True)

# === LOAD HISTORICAL DATA ===
hist_files = pd.concat([
    pd.read_excel(os.path.join(hist_path, file)).assign(source=file)
    for file in os.listdir(hist_path) if file.endswith(".xlsx")
])

df = hist_files.rename(columns={
    'Date': 'stay_date',
    'Paying Room (Room Sold) Today Actual': 'rooms_occupied',
    'RNA Today': 'available_rooms'
})
df['stay_date'] = pd.to_datetime(df['stay_date'])
df['occupancy'] = df['rooms_occupied'] / df['available_rooms']
df['month'] = df['stay_date'].dt.month
df['dow'] = df['stay_date'].dt.dayofweek
df['is_weekend'] = df['stay_date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int)

le = LabelEncoder()
df['dow_encoded'] = le.fit_transform(df['stay_date'].dt.day_name())

# === FEATURES ===
features = ['month', 'is_weekend', 'dow_encoded']
X = df[features]
y = df['occupancy']

# === RFE SETUP ===
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

rfe = RFECV(estimator=model, step=1, cv=cv, scoring='neg_mean_squared_error')
rfe.fit(X, y)

# === OUTPUT RESULTS ===
ranking = pd.DataFrame({
    'feature': features,
    'ranking': rfe.ranking_,
    'selected': rfe.support_
}).sort_values(by='ranking')

print("\n🧠 Feature Ranking via RFE (1 = most important):")
print(ranking)

# === OPTIONAL: Plot scores ===
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(rfe.cv_results_['mean_test_score']) + 1), rfe.cv_results_['mean_test_score'], marker='o')
plt.xlabel("Number of Features Selected")
plt.ylabel("Cross-Validated Score (Neg. MSE)")
plt.title("RFECV Performance")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "xgb_rfe_cv_score_chart.png"))
plt.show()

# === Save results to Excel ===
ranking.to_excel(os.path.join(output_path, "xgb_rfe_feature_ranking.xlsx"), index=False)
print("✅ RFE results saved to xgb_rfe_feature_ranking.xlsx")
