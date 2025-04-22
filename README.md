# FluxRate
**AI-Powered Pricing & Forecasting Engine for Budget Hotels**  
_No RMS. No Bloat. Just Pure Revenue Intelligence._

FluxRate is a lightweight, Python-based revenue optimization tool designed for budget hotels that operate without traditional Revenue Management Systems (RMS). Built for ease, speed, and precision, FluxRate empowers revenue teams with powerful machine learning forecasts and pricing recommendations — all without the overhead of commercial RMS platforms like IDeaS or Duetto.

---

## Key Features

- ✅ **Forecasting Models**: Poisson, XGBoost, k-Nearest Neighbor (k-NN), and Soft Voting Ensemble
- ✅ **Dynamic Pricing**: Real-time BAR level recommendations based on forecasted demand and compset analysis
- ✅ **Same-Day Pickup Tracker**: Automates daily occupancy pickup calculations
- ✅ **LOS Trend Analysis**: Visual length-of-stay behavior over a rolling window
- ✅ **Streamlit Dashboard**: Interactive UI for monitoring forecasts, pricing actions, and model accuracy (MAPE)
- ✅ **Built-in Flexibility**: Designed for hotels using OTA channels, without any RMS or data science staff

---

## Target Users
- Budget hotels & independent properties
- Revenue managers and hotel operators without RMS tools
- Hospitality teams looking for low-cost, high-impact forecasting

---

## Technology Stack

- Python 3.10+
- Pandas, NumPy, Scikit-learn
- Streamlit (for dashboard)
- Excel I/O: `openpyxl`, `xlrd`
- Forecasting Models: Poisson Regression, XGBoost, k-NN, Soft Voting Ensemble

---

## Sample Folder Structure

FluxRate/ ├── scripts/ # Python scripts for forecasting, pricing, and pickup ├── dashboard/ # Streamlit UI components ├── data/ # Sample OTB, rate shopping, and forecast files ├── output/ # Generated reports: pricing, MAPE, pickup logs ├── README.md ├── requirements.txt └── .gitignore

---

## Developed By

**Chris Legaspi**  
Chief Commercial Officer – Archipelago International  
Executive MBA Candidate – Asian Institute of Management (EMBA 2025)
