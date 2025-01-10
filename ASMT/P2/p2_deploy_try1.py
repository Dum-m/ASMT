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

# Correct model filenames based on your local version
MODEL_FILES = {
    'PHEV': {
        0: 'CEM_tf0_PHEV_20241231005132.pkl',
        1: 'CEM_tf1_PHEV_20241231005142.pkl',
        2: 'CEM_tf2_PHEV_20241231005106.pkl'
    },
    'BEV': {
        0: 'CEM_tf0_BEV_20241231005043.pkl',
        1: 'CEM_tf1_BEV_20241231005052.pkl',
        2: 'CEM_tf2_BEV_20241231005019.pkl'
    }
}

def get_file_path(filename):
    """Get the correct file path in different environments"""
    try:
        # Try different possible locations
        possible_paths = [
            filename,  # Current directory
            os.path.join(os.path.dirname(__file__), filename),  # Script directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),  # Absolute script directory
            os.path.join('models', filename)  # Models subdirectory
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        st.error(f"Could not find {filename} in any expected location")
        return None
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
                model_file = MODEL_FILES[ev_type][cluster]
                model_path = get_file_path(model_file)
                
                if model_path is None:
                    continue
                    
                try:
                    # Load PyTorch Tabular model with proper error handling
                    model = torch.load(model_path, map_location=torch.device('cpu'))
                    models[ev_type][cluster] = model
                except Exception as e:
                    st.error(f"Error loading model {model_file}: {str(e)}")
                    # Try alternative loading method
                    try:
                        model = joblib.load(model_path)
                        models[ev_type][cluster] = model
                    except Exception as e2:
                        st.error(f"Alternative loading failed for {model_file}: {str(e2)}")
                        
        return models
    except Exception as e:
        st.error(f"Error in model loading process: {str(e)}")
        return None

# Streamlit interface
st.title("2024 EV Prediction by ZCTA5 in New York State")

# Load models
with st.spinner("Loading models..."):
    models = load_models()
    if models is None:
        st.error("Failed to load models. Please check the model files and their locations.")
        st.stop()

# Add file location debugging if needed
if st.checkbox("Show debug information"):
    st.write("Current working directory:", os.getcwd())
    st.write("Script location:", os.path.dirname(os.path.abspath(__file__)))
    st.write("Available files:", os.listdir())

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
                    
                    # Handle prediction based on model type
                    if isinstance(model, TabularModel):
                        prediction = model.predict(row_df)
                    else:
                        prediction = model.predict(row_df)
                    
                    # Handle different prediction formats
                    if isinstance(prediction, pd.DataFrame):
                        predicted_value = prediction.iloc[0, 0]
                    elif isinstance(prediction, pd.Series):
                        predicted_value = prediction.iloc[0]
                    elif isinstance(prediction, (np.ndarray, list)):
                        predicted_value = prediction[0]
                    else:
                        predicted_value = float(prediction)
                    
                    return int(round(predicted_value))
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    if st.checkbox("Show error details"):
                        st.write("Error type:", type(e))
                        st.write("Error message:", str(e))
                    return None
            else:
                st.error("Model not available for this traffic cluster.")
                return None

        with col1:
            if st.button("Predict PHEV"):
                with st.spinner("Predicting PHEV..."):
                    result = predict_ev('PHEV')
                    if result is not None:
                        st.success(f"2024 PHEV in ZCTA5 {zcta_selected} will be: {result}")

        with col2:
            if st.button("Predict BEV"):
                with st.spinner("Predicting BEV..."):
                    result = predict_ev('BEV')
                    if result is not None:
                        st.success(f"2024 BEV in ZCTA5 {zcta_selected} will be: {result}")

    except Exception as e:
        st.error(f"Error processing selection: {str(e)}")
