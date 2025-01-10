import streamlit as st
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, AutoIntConfig, TabNetModelConfig, FTTransformerConfig,DANetConfig,GANDALFConfig,GatedAdditiveTreeEnsembleConfig,MDNConfig

from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)

import matplotlib.pyplot as plt



# Load data and models
df = pd.read_csv(Path(__file__).parent /'df_fordep2022.csv')
df['zcta20'] = df['zcta20'].fillna(0).astype(int).astype(str).str.zfill(5)



models = {
    'PHEV': {
        0: joblib.load(Path(__file__).parent /'CEM_tf0_PHEV_20241231005132.pkl'),
        1: joblib.load(Path(__file__).parent /'CEM_tf1_PHEV_20241231005142.pkl'),
        2: joblib.load(Path(__file__).parent /'CEM_tf2_PHEV_20241231005106.pkl')
    },
    'BEV': {
        0: joblib.load(Path(__file__).parent /'CEM_tf0_BEV_20241231005043.pkl'),
        1: joblib.load(Path(__file__).parent /'CEM_tf1_BEV_20241231005052.pkl'),
        2: joblib.load(Path(__file__).parent /'CEM_tf2_BEV_20241231005019.pkl')
    }
}

def format_prediction(prediction):
    try:
        if isinstance(prediction, (pd.DataFrame, pd.Series)):
            return f"{prediction.iloc[0]:.2f}"
        elif isinstance(prediction, np.ndarray):
            return f"{prediction[0]:.2f}"
        else:
            return f"{float(prediction):.2f}"
    except Exception as e:
        st.error(f"Error formatting prediction: {str(e)}")
        return "Error"


# Streamlit app


st.title("2024 EV Prediction by ZCTA5 in New York State")

# Dropdown for ZCTA20 selection
zcta_selected = st.selectbox("Select your ZCTA5:", df['zcta20'].unique())

# Get the row of the selected ZCTA20
if zcta_selected:
    row = df[df['zcta20'] == zcta_selected].iloc[0]
    traffic_cluster3 = row['traffic_cluster3']

# Buttons for predictions
col1, col2 = st.columns(2)
with col1:
    if st.button("Predict PHEV"):
        model = models['PHEV'].get(traffic_cluster3)
        if model:
            # Convert the numpy array back to a DataFrame with original column names
            row_df = pd.DataFrame(row.values.reshape(1, -1), columns=df.columns)
            prediction = model.predict(row_df)
            # Handle prediction output types
            if isinstance(prediction, pd.DataFrame):
                predicted_value = prediction.iloc[0, 0]
            elif isinstance(prediction, pd.Series):
                predicted_value = prediction.iloc[0]
            elif isinstance(prediction, (np.ndarray, list)):
                predicted_value = prediction[0]
            else:
                st.error("Unexpected prediction output format.")
                predicted_value = None
            
            if predicted_value is not None:
                predicted_value = int(round(predicted_value))
                st.success(f"2024 PHEV in ZCTA5 {zcta_selected} will be: {predicted_value}")
        else:
            st.error("Model not available for this traffic cluster.")



with col2:
    if st.button("Predict BEV"):
        model = models['BEV'].get(traffic_cluster3)
        if model:
            # Convert row to DataFrame for model compatibility
            row_df = pd.DataFrame(row.values.reshape(1, -1), columns=df.columns)
            prediction = model.predict(row_df)
            
            # Handle prediction output types
            if isinstance(prediction, pd.DataFrame):
                predicted_value = prediction.iloc[0, 0]
            elif isinstance(prediction, pd.Series):
                predicted_value = prediction.iloc[0]
            elif isinstance(prediction, (np.ndarray, list)):
                predicted_value = prediction[0]
            else:
                st.error("Unexpected prediction output format.")
                predicted_value = None
            
            if predicted_value is not None:
                predicted_value = int(round(predicted_value))
                st.success(f"2024 BEV in ZCTA5 {zcta_selected} will be: {predicted_value}")
        else:
            st.error("Model not available for this traffic cluster.")




