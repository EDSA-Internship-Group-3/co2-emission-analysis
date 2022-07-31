# Script dependencies
# Libraries for Anaysis
import numpy as np
import pandas as pd

import streamlit as st

# Libraries for Plotting Analysis
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import time                                             # For Calulating ALgo Time Run

# Libraries to Save/Restore Models
import pickle



# -------------------------------------------------------------------
    #        Modeling
# -------------------------------------------------------------------
def xgboost_model():

    # We make use of an StandardScaler used.
    standard_scaler = pickle.load(
        open('resources/models/220729_standard_scaler.pkl', 'rb')
        )

    # We make use of an xgboost model trained on .
    model = pickle.load(
        open('resources/models/220729_transformed_xgb.pkl', 'rb')
        )


    df_2 = pd.read_feather("resources/datasets/220714_full_dataset.feather")

    st.dataframe(df_2.sample(n=8))

    # Importing data
    df = pd.read_csv("resources/datasets/Our_CO2emission_Modelling_Data.csv")

    row = df.sample(n=1)

    actual_val = row['CO2_emission'].values[0]
        
    
    # Drop Unamed Column & predict target
    predict_features = row.drop(['Unnamed: 0',"CO2_emission"], axis=1)


    # convert the scaled predictor values into a dataframe
    scaled_predict_features = standard_scaler.transform(predict_features)
    predict_val = model.predict(scaled_predict_features)[0]
    # predict_val = standard_scaler.inverse_transform(pred_val.reshape(-1,1))

    st.write("Predicting for the following data point:")
    st.dataframe(row)

    st.text(f"Actual Value: \t\t{actual_val}")
    st.text(f"Predicted Value: \t{predict_val}")


    # ---------------------------------------------------------
    ### Remodelling with Essential Features ###
    # ---------------------------------------------------------



    # Create Xgboost Model

    return 


    """load model
    select rand clumn in data
    prepare data to predict ie.
    drop unneeded columns
    scaling

    """





