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

@st.experimental_memo
def load_resources():
    # We make use of an StandardScaler used.
    standard_scaler = pickle.load(
        open('resources/models/220729_standard_scaler.pkl', 'rb')
        )
    model = pickle.load(
        open('resources/models/220729_transformed_xgb.pkl', 'rb')
        )

    E_TYPE_DICT = {
        0:'Renewable',1:'Nuclear',2:'Natural Gas',
        3:'Petroleum',4:'Coal',5:'All',
    }

    # Importing data
    df = pd.read_feather("resources/datasets/220801_data.feather")

    return df, model,standard_scaler,E_TYPE_DICT


@st.experimental_memo
def draw_pred_chart(df,row, predict_val):

    fig = go.Figure()

    fig.add_vline(x=row['Year'].values[0],
        line_dash='dash',line_width=1,
        line_color='yellow')
    fig.add_trace(
        go.Scatter(
            x=df.loc[(df.Country==row['Country'].values[0])&
                (df.e_type==row['e_type'].values[0])]
                .groupby('Year',as_index=False)
                .agg({'CO2_emission':'sum'})
                .loc[:,'Year'],
            y=df.loc[(df.Country==row['Country'].values[0])&
                (df.e_type==row['e_type'].values[0])]
                .groupby('Year',as_index=False)
                .agg({'CO2_emission':'sum'})
                .loc[:,'CO2_emission'],
            name=f"CO2 Emission plot."
        )
    )
    fig.add_trace(
        go.Scatter(
            x=row['Year'].values,
            y=[predict_val],
            name=f"Predicted Value"
        )
    )
    print(row['Year'].values,)
    return fig

@st.experimental_memo
def predict_value(x_features):
    features = x_features[['e_type', 'e_con', 'e_prod', 'GDP', 'Population', 'ei_capita', 'ei_gdp', 'pop_growth', 'pop_density', 'Manuf_GDP', 'Agric_GDP', 'Deforestation', 'emission_per_cap']]
    _, model, standard_scaler, _ = load_resources()
    scaled_features = standard_scaler.transform(features)
    predict_val = model.predict(scaled_features)[0]
    
    return predict_val



# -------------------------------------------------------------------
    #        Modeling
# -------------------------------------------------------------------
def xgboost_model():

    df, model, standard_scaler, E_TYPE_DICT = load_resources()




    # We make use of an xgboost model trained on .
    st.dataframe(df.sample(n=4))

    st.markdown("---")
    if st.button("Select Random data point"):
        row = df.loc[df['CO2_emission']>1].sample(n=1)
    else:
        row = df.loc[df['CO2_emission']>1].sample(n=1)


    actual_val = row['CO2_emission'].values[0]


    predict_val = predict_value(row)


    

    col1, col2 = st.columns([2,4])
    with col1:
        st.write("Predicting for the following data point:")
        st.dataframe(row)

        st.text(f"Actual Value: \t\t{actual_val}")
        st.text(f"Predicted Value: \t{predict_val:.2f}")
        perc_diff = abs(predict_val - actual_val)/actual_val*100
        st.text(f"Percentage Diff: \t{perc_diff:.1f}%")


    fig = draw_pred_chart(df, row, predict_val)

    fig.update_layout(
        title=f"""
{row['Country'].values[0]}'s CO2 Emission plot for \
{E_TYPE_DICT[row['e_type'].values[0]]} Energy sources.
        """,
        margin=dict(l=0,r=0,t=25,b=50)
    )
    fig.update_xaxes(showgrid=False)

    with col2:
        st.plotly_chart(fig)



    col1, col2 = st.columns([2,4])

    with col1:
        st.markdown('### Providing future predictions')
        st.markdown('Use this section to make future predictions.')
        pred_country = st.selectbox('Select Country:',options=df["Country"].unique())

        pred_e_type = st.selectbox('Select Energy Type:',options=E_TYPE_DICT.values())

        df = df.loc[(df['Country']==pred_country)&
        (df['e_type']==list(E_TYPE_DICT.keys())[list(E_TYPE_DICT.values()).index(pred_e_type)])]

        pred_e_type = list(E_TYPE_DICT.keys())[list(E_TYPE_DICT.values()).index(pred_e_type)]

        last_row = df.loc[(df['Year']==df['Year'].max())]

        pred_year = st.slider("Select year:",
            min_value = int(last_row['Year'].values[0]) + 1,
            max_value = 2050,
            )
        pred_gdp = st.slider('Annual GDP Growth (%)',
            min_value=-0.5,
            max_value=0.5,
            value=0.0,
            step=0.05,
            )
        pred_e_con = st.slider('Change in Energy Consumption (%)',
            min_value=-0.5,
            max_value=0.5,
            value=0.0,
            step=0.05,
            )
        pred_e_prod = st.slider('Change in Energy Production (%)',
            min_value=-0.5,
            max_value=0.5,
            value=0.0,
            step=0.05,
            )
        pred_pop_growth = st.slider('Annual Population Growth (%)',
            min_value=df['pop_growth'].min(),
            max_value=df['pop_growth'].max()
            )

    time_factor = pred_year - last_row['Year'].values[0]

    pred_gdp = last_row['GDP'].values[0]*(1 + pred_gdp)**time_factor
    pred_e_con = last_row['e_con'].values[0]*(1 + pred_e_con)**time_factor
    pred_e_prod = last_row['e_prod'].values[0]*(1 + pred_e_prod)**time_factor
    pred_pop = last_row['Population'].values[0]*(1 + pred_pop_growth)**time_factor

    pred_row = pd.DataFrame(
        [{  
            'Country': pred_country,
            'e_type': pred_e_type,
            'Year': pred_year,
            'e_con': pred_e_con,
            'e_prod': pred_e_prod,
            'GDP': pred_gdp,
            'Population': pred_pop,
            'ei_capita': last_row['ei_capita'].values[0],
            'ei_gdp': last_row['ei_gdp'].values[0],
            'pop_growth': pred_pop_growth,
            'pop_density': last_row['pop_density'].values[0],
            'Manuf_GDP': last_row['Manuf_GDP'].values[0],
            'Agric_GDP': last_row['Agric_GDP'].values[0],
            'Continent': last_row['Continent'].values[0],
            'Deforestation': last_row['Deforestation'].values[0],
            'emission_per_cap': last_row['emission_per_cap'].values[0],
        }]
    )

    predict_val = predict_value(pred_row)
    with col2:
        fig = draw_pred_chart(df, pred_row, predict_val)

        st.plotly_chart(fig)


    print(predict_val)


    # ---------------------------------------------------------
    ### Remodelling with Essential Features ###
    # ---------------------------------------------------------



    # Create Xgboost Model


    """load model
    select rand clumn in data
    prepare data to predict ie.
    drop unneeded columns
    scaling

    """





