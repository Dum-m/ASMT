
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib  # Import joblib for saving models
import os  # For handling directory paths
from pathlib import Path




# best_model_name='Random_Forest'

model_path = Path(__file__).parent / 'Linear_Regression.pkl'

model = joblib.load(model_path)


feature_limits = {
    'NO2': (0.003200, 0.009600),  # min, max
    'O3': (0.013000, 0.024000),
    'PM 10': (16.840000, 64.110000),
    'PM 2.5': (10.020000, 52.860000),
    'SO2': (0.000900, 0.002800)
}


st.title("CO Predicition")



no2 = st.number_input("Enter NO2 concentration (ppm):", min_value=feature_limits['NO2'][0], max_value=feature_limits['NO2'][1], step=0.0001, format="%.4f")
o3 = st.number_input("Enter O3 concentration (ppm):", min_value=feature_limits['O3'][0], max_value=feature_limits['O3'][1], step=0.0001, format="%.4f")
pm10 = st.number_input("Enter PM 10 concentration (µg/m³):", min_value=feature_limits['PM 10'][0], max_value=feature_limits['PM 10'][1], step=0.1, format="%.1f")
pm25 = st.number_input("Enter PM 2.5 concentration (µg/m³):", min_value=feature_limits['PM 2.5'][0], max_value=feature_limits['PM 2.5'][1], step=0.1, format="%.1f")
so2 = st.number_input("Enter SO2 concentration (ppm):", min_value=feature_limits['SO2'][0], max_value=feature_limits['SO2'][1], step=0.0001, format="%.4f")


input_data = {'NO2': no2, 'O3': o3, 'PM 10': pm10, 'PM 2.5': pm25, 'SO2': so2}


def validate_input(input_dict):
    for feature, value in input_dict.items():
        min_val, max_val = feature_limits[feature]
        if value < min_val or value > max_val:
            return f"Error: {feature} must be between {min_val} and {max_val}."
    return None


validation_error = validate_input(input_data)

if validation_error:
    st.error(validation_error)
else:

    input_array = np.array([[no2, o3, pm10, pm25, so2]])

    if st.button("Predict CO concentration"):
        try:
            prediction = model.predict(input_array)
            st.success(f"Predicted CO concentration: {prediction[0]:.2f} ppm")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
