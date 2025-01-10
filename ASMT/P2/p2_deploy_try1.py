import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)

# Configure Streamlit page
st.set_page_config(
    page_title="EV Prediction NYS",
    page_icon="🚗",
    layout="wide"
)

# Function to get absolute path for files
def get_file_path(filename):
    try:
        # First try the local directory
        if os.path.exists(filename):
            return filename
        # Then try the parent directory of the script
        parent_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(parent_path, filename)
        if os.path.exists(file_path):
            return file_path
        # If neither exists, raise an error
        raise FileNotFoundError(f"Could not find {filename}")
    except Exception as e:
        st.error(f"Error accessing file {filename}: {str(e)}")
        return None

# Load data with error handling
try:
    data_file = get_file_path('df_fordep2022.csv')
    if data_file is None:
        st.error("Could not load data file. Please check if df_fordep2022.csv exists in the application directory.")
        st.stop()
    
    df = pd.read_csv(data_file)
    df['zcta20'] = df['zcta20'].fillna(0).astype(int).astype(str).str.zfill(5)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Load models with error handling
@st.cache_resource
def load_models():
    models = {
        'PHEV': {},
        'BEV': {}
    }
    try:
        for ev_type in ['PHEV', 'BEV']:
            for cluster in [0, 1, 2]:
                model_file = f'CEM_tf{cluster}_{ev_type}_20241231005{cluster:03d}.pkl'
                model_path = get_file_path(model_file)
                if model_path is None:
                    st.error(f"Could not load model file: {model_file}")
                    continue
                models[ev_type][cluster] = joblib.load(model_path)
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Load models
models = load_models()
if models is None:
    st.error("Failed to load models. Please check if model files exist in the application directory.")
    st.stop()

# Streamlit app interface
st.title("2024 EV Prediction by ZCTA5 in New York State")

# Dropdown for ZCTA20 selection
zcta_selected = st.selectbox("Select your ZCTA5:", df['zcta20'].unique())

if zcta_selected:
    try:
        row = df[df['zcta20'] == zcta_selected].iloc[0]
        traffic_cluster3 = row['traffic_cluster3']
        
        col1, col2 = st.columns(2)
        
        def predict_ev(ev_type):
            model = models[ev_type].get(traffic_cluster3)
            if model:
                try:
                    row_df = pd.DataFrame(row.values.reshape(1, -1), columns=df.columns)
                    prediction = model.predict(row_df)
                    
                    # Handle different prediction formats
                    if isinstance(prediction, pd.DataFrame):
                        predicted_value = prediction.iloc[0, 0]
                    elif isinstance(prediction, pd.Series):
                        predicted_value = prediction.iloc[0]
                    elif isinstance(prediction, (np.ndarray, list)):
                        predicted_value = prediction[0]
                    else:
                        raise ValueError("Unexpected prediction format")
                    
                    return int(round(predicted_value))
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    return None
            else:
                st.error("Model not available for this traffic cluster.")
                return None

        with col1:
            if st.button("Predict PHEV"):
                result = predict_ev('PHEV')
                if result is not None:
                    st.success(f"2024 PHEV in ZCTA5 {zcta_selected} will be: {result}")

        with col2:
            if st.button("Predict BEV"):
                result = predict_ev('BEV')
                if result is not None:
                    st.success(f"2024 BEV in ZCTA5 {zcta_selected} will be: {result}")

    except Exception as e:
        st.error(f"Error processing selection: {str(e)}")
