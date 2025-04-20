# daily_forecast_runner.py

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# === PATH SETUP ===
base_path = os.path.expanduser("~/Documents/2025_CAPSTONE")
output_path = os.path.join(base_path, "output")
pickup_path = os.path.join(base_path, "data/pickup")
otb_path = os.path.join(base_path, "data/otb")
hist_path = os.path.join(base_path, "data/historical")
os.makedirs(output_path, exist_ok=True)

# === TIMESTAMP FOR FILES ===
timestamp = datetime.now().strftime('%Y%m%d_%H%M')

# === YOUR FORECAST CODE GOES HERE ===
# - load pickup files
# - load OTB file
# - load historicals
# - train model
# - generate forecast
# - export .xlsx + .png

# (I'll help you plug in each section if you say go)

print(f"[✅] Report generated at {timestamp}")