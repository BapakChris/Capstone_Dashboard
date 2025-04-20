import streamlit as st
import pandas as pd
import os
from datetime import datetime, date, timedelta
import altair as alt
import io

# === CONFIG ===
base_path = os.path.expanduser("~/Documents/2025_CAPSTONE")
output_path = "output"
hotel_name = "Neo Puri Indah"

# === LOAD FORECAST + ACCURACY ===
def load_latest(prefix):
    st.write("📁 Streamlit sees these files in /output:", os.listdir(output_path))
    st.write("🔍 Looking for prefix:", prefix)
    files = [f for f in os.listdir(output_path) if f.startswith(prefix) and f.endswith(".xlsx")]
    if not files:
        return None
    latest_file = sorted(files, key=lambda x: os.path.getmtime(os.path.join(output_path, x)))[-1]
    df = pd.read_excel(os.path.join(output_path, latest_file))

    # Normalize column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Convert stay_date if it exists
    if 'stay_date' in df.columns:
        df['stay_date'] = pd.to_datetime(df['stay_date'])

    return df

st.write("Files in /output:", os.listdir("output"))
forecast_df = load_latest("forecast_soft_ensemble_quality")
forecast_df['stay_date'] = pd.to_datetime(forecast_df['stay_date']).dt.date
accuracy_df = load_latest("forecast_soft_ensemble_accuracy")

# === STREAMLIT UI ===
st.set_page_config(page_title="Occupancy Forecast Dashboard", layout="wide")
st.title(hotel_name)
st.subheader("Occupancy Forecast Dashboard")
st.markdown(
    f"<div style='margin-top:-15px; font-size: 0.9em; color: gray;'>"
    f"Forecast as of {date.today().strftime('%B %d, %Y')} | Prepared by Chris Legaspi</div>",
    unsafe_allow_html=True
)
st.markdown("---")

if forecast_df is None:
    st.warning("No forecast file found.")
    st.stop()

forecast_df['stay_date'] = pd.to_datetime(forecast_df['stay_date']).dt.date
forecast_df['date_type'] = forecast_df['stay_date'].apply(
    lambda x: "Past" if x < date.today() else "Future"
)

past_df = forecast_df[forecast_df['stay_date'] < date.today()]
future_df = forecast_df[forecast_df['stay_date'] >= date.today()]

# === CHART FUNCTION ===
def build_chart(data):
    return alt.Chart(data).mark_line(point=True).encode(
        x=alt.X('stay_date:T', title='Stay Date'),
        y=alt.Y('ensemble_forecast:Q', title='Occupancy (%)', scale=alt.Scale(domain=[0, 100])),
        tooltip=['stay_date:T', 'ensemble_forecast:Q']
    ).properties(width=800, height=300)

# === PAST CHART ===
if not past_df.empty:
    st.subheader("Historical Forecast Performance")
    st.altair_chart(build_chart(past_df))

# === FORECAST CHART + SLIDER ===
st.subheader("Forecasted Occupancy (Today Onward)")

min_date = forecast_df['stay_date'].min()
max_date = forecast_df['stay_date'].max()

date_range = st.slider(
    "Select forecast date range:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    label_visibility="visible"
)

filtered_future = forecast_df[
    (forecast_df['stay_date'] >= date_range[0]) &
    (forecast_df['stay_date'] <= date_range[1])
]

st.altair_chart(build_chart(filtered_future))

# === BAR CHART: Avg Occupancy by DOW ===
st.subheader("Average Occupancy by Day of the Week")
dow_df = filtered_future.copy()
dow_df['day_name'] = pd.to_datetime(dow_df['stay_date']).dt.day_name()
dow_summary = dow_df.groupby('day_name')['ensemble_forecast'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
).reset_index()

bar_chart = alt.Chart(dow_summary).mark_bar().encode(
    x=alt.X('day_name:N', title='Day of Week', sort=list(dow_summary['day_name'])),
    y=alt.Y('ensemble_forecast:Q', title='Average Occupancy (%)'),
    tooltip=['day_name:N', 'ensemble_forecast:Q']
).configure_axisX(labelAngle=45).properties(width=700, height=300)

st.altair_chart(bar_chart)

# === REAL HORIZONTAL BAR: Avg Same-Day Pickup by DOW ===
pickup_log_path = os.path.join(output_path, "same_day_pickup_dow_log.xlsx")
if os.path.exists(pickup_log_path):
    df_pickup = pd.read_excel(pickup_log_path)
    df_pickup['stay_date'] = pd.to_datetime(df_pickup['stay_date']).dt.date
    df_pickup['day_of_week'] = df_pickup['day_of_week'].astype(str)

    dow_pickup_summary = df_pickup.groupby('day_of_week')['pickup_rooms'].agg(['count', 'mean']).reset_index()
    dow_pickup_summary = dow_pickup_summary.rename(columns={'count': 'sample_size', 'mean': 'avg_pickup_rooms'})
    dow_pickup_summary['avg_pickup_rooms'] = dow_pickup_summary['avg_pickup_rooms'].round(2)

    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_pickup_summary['day_of_week'] = pd.Categorical(dow_pickup_summary['day_of_week'], categories=weekday_order, ordered=True)
    dow_pickup_summary = dow_pickup_summary.sort_values('day_of_week')

    st.subheader("Average Same-Day Pickup by Day of the Week (Actual Rooms)")
    pickup_chart = alt.Chart(dow_pickup_summary).mark_bar().encode(
        y=alt.Y('day_of_week:N', title='Day of Week', sort=weekday_order),
        x=alt.X('avg_pickup_rooms:Q', title='Avg Pickup (Rooms)'),
        tooltip=['day_of_week:N', 'avg_pickup_rooms:Q', 'sample_size:Q']
    ).properties(width=700, height=300)
    st.altair_chart(pickup_chart)
else:
    st.info("Same-day pickup log not found.")

# === LOS TREND CHART ===
los_path = os.path.join(output_path, "los_trend.xlsx")
if os.path.exists(los_path):
    df_los = pd.read_excel(los_path)
    df_los['stay_date'] = pd.to_datetime(df_los['stay_date'])

    st.subheader("Length of Stay Trend based on pickup")

    min_los_date = df_los['stay_date'].min().date()
    max_los_date = df_los['stay_date'].max().date()
    los_date_range = st.slider(
        "Select LOS trend date range:",
        min_value=min_los_date,
        max_value=max_los_date,
        value=(min_los_date, max_los_date),
        label_visibility="visible"
    )

    df_los_filtered = df_los[
        (df_los['stay_date'].dt.date >= los_date_range[0]) &
        (df_los['stay_date'].dt.date <= los_date_range[1])
    ]

    avg_los_display = df_los_filtered['LOS'].mean().round(2)
    st.markdown(f"<div style='text-align: right; font-size: 0.9em; color: gray;'>Average LOS = {avg_los_display} nights</div>", unsafe_allow_html=True)

    los_chart = alt.Chart(df_los_filtered).mark_line(point=True).encode(
        x=alt.X('stay_date:T', title='Stay Date'),
        y=alt.Y('LOS:Q', title='Avg Length of Stay (nights)'),
        tooltip=['stay_date:T', 'LOS:Q']
    ).properties(width=800, height=300)

    st.altair_chart(los_chart)
else:
    st.info("LOS trend file not found.")

# === FORECAST ACCURACY TABLE (MAPE) ===
accuracy_files = [f for f in os.listdir(output_path) if f.startswith("forecast_soft_ensemble_accuracy_") and f.endswith(".xlsx")]
accuracy_df = None

if accuracy_files:
    latest_accuracy_file = sorted(accuracy_files, key=lambda x: os.path.getmtime(os.path.join(output_path, x)))[-1]
    accuracy_df = pd.read_excel(os.path.join(output_path, latest_accuracy_file))

if accuracy_df is not None:
    accuracy_df['stay_date'] = pd.to_datetime(accuracy_df['stay_date'])
    accuracy_valid = accuracy_df[accuracy_df['ensemble'].notnull()]

    if not accuracy_valid.empty:
        st.subheader("Forecast Accuracy (MAPE)")
        accuracy_valid['stay_date'] = accuracy_valid['stay_date'].dt.strftime('%Y-%m-%d')
        display_df = accuracy_valid.rename(columns={
            'stay_date': 'Stay Date',
            'occupancy': 'Occupancy',
            'poisson': 'Poisson Model',
            'xgb': 'XGBoost Model',
            'knn': 'kNN Model',
            'ensemble': 'Ensemble',
            'mape': 'MAPE'
        })

        st.dataframe(
            display_df[['Stay Date', 'Occupancy', 'Poisson Model', 'XGBoost Model', 'kNN Model', 'Ensemble', 'MAPE']],
            use_container_width=True,
            hide_index=True
        )

# === BAR LEVEL RECOMMENDATION TABLE (MERGED) ===
st.subheader("BAR Level Recommendations – Next 7 Days")

bar_file = load_latest("bar_level_predictions")
pricing_file = load_latest("pricing_recommendation_weighted")

if bar_file is not None and pricing_file is not None:
    bar_file.columns = [col.strip().lower().replace(" ", "_") for col in bar_file.columns]
    pricing_file.columns = [col.strip().lower().replace(" ", "_") for col in pricing_file.columns]

    bar_file['stay_date'] = pd.to_datetime(bar_file['stay_date']).dt.date
    pricing_file['stay_date'] = pd.to_datetime(pricing_file['stay_date']).dt.date

    merged = bar_file.merge(
        pricing_file[['stay_date', 'bar_rate', 'rate_decision_reason']],
        on='stay_date',
        how='left'
    )
    merged['recommendation_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    next_7 = merged[
        (merged['stay_date'] >= date.today()) &
        (merged['stay_date'] <= date.today() + timedelta(days=6))
    ]

    if not next_7.empty:
        next_7['bar_rate'] = pd.to_numeric(next_7['bar_rate'], errors='coerce')
        next_7['ensemble_forecast'] = pd.to_numeric(next_7['ensemble_forecast'], errors='coerce')

        display_cols = [
            'stay_date', 'days_to_arrival', 'ensemble_forecast',
            'recommended_bar_level_(weighted)', 'predicted_bar_level_(rf)', 'predicted_bar_level_(xgb)',
            'bar_rate', 'rate_decision_reason', 'recommendation_timestamp'
        ]

        st.dataframe(next_7[display_cols].style.format({
            'ensemble_forecast': '{:.2%}',
            'bar_rate': '{:,.0f}'
        }))
    else:
        st.info("No BAR recommendations available for the next 7 days.")
else:
    st.warning("Required prediction files not found.")

st.markdown(
    "<div style='text-align:right; color:gray; font-size:0.85em;'>"
    "🖨️ To export: Right-click → Print or press ⌘+P (Mac) / Ctrl+P (Windows) → Save as PDF"
    "</div>",
    unsafe_allow_html=True
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
