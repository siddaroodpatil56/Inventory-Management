import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('features_columns.pkl')

st.title("Inventory Risk Prediction App")
st.write("Predict whether an inventory item is at risk of Stockout or Overstock using Logistic Regression.")

st.subheader("Enter Inventory Parameters")

# Inputs matching our daily data
current_stock = st.number_input("Current Stock", min_value=0.0, max_value=10000.0, value=501.0, step=1.0)
avg_daily_demand = st.number_input("Average Daily Demand", min_value=0.10, max_value=10000.00,value=49.83, step=0.1)
lead_time_days = st.number_input("Lead Time (Days)", min_value=0.25, max_value=20.00, value=7.02, step=0.1)
reorder_point = st.number_input("Reorder Point", min_value=150.18, max_value=1000.95, value=350.96, step=0.1)
sales_last_30_days = st.number_input("Sales Last 30 Days", min_value=400.47, max_value=5000.41, value=1497.60, step=1.0)
stock_turnover_ratio = st.number_input("Stock Turnover Ratio", min_value=0.10, max_value=10.0, value=4.00, step=0.01)
forecast_error = st.number_input("Forecast Error", min_value=-100.0, max_value=100.00, value=-0.12, step=0.01)

# Categorical inputs matching the manual binary mapping
season = st.selectbox("Season", options=["Off-Peak (0)", "Peak (1)"])
item_category = st.selectbox("Item Category", options=["Non-Essential (0)", "Essential (1)"])
supplier_reliability = st.selectbox("Supplier Reliability", options=["Low/Medium (0)", "High (1)"])

# Converstion of categorical selections to numerical values
season_val = 1 if "Peak (1)" in season else 0
category_val = 1 if "Essential (1)" in item_category else 0
supplier_val = 1 if "High (1)" in supplier_reliability else 0

if st.button("Predict Stock Status"):
    input_data = {
        'current_stock': [current_stock],
        'avg_daily_demand': [avg_daily_demand],
        'lead_time_days': [lead_time_days],
        'reorder_point': [reorder_point],
        'sales_last_30_days': [sales_last_30_days],
        'stock_turnover_ratio': [stock_turnover_ratio],
        'forecast_error': [forecast_error],
        'season': [season_val],
        'item_category': [category_val],
        'supplier_reliability': [supplier_val]
    }

    input_df = pd.DataFrame(input_data)
    input_df = input_df[feature_columns]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("Status: **Overstock**")
    else:
        st.error("Status: **Stockout Risk**")

    st.write(f"Confidence: {np.max(probability)*100:.2f}%")
