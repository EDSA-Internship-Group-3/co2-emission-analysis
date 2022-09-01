# -------------------------------------------------------------------
#        DEFINES THE DATA ANALYSIS SECTION
# -------------------------------------------------------------------
import streamlit as st

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import pandas as pd

@st.experimental_memo
def load_resources():
    E_TYPE_DICT = {
        0:'Renewable',1:'Nuclear',2:'Natural Gas',
        3:'Petroleum',4:'Coal',5:'All',
    }

    # Importing data
    df = pd.read_feather("resources/datasets/220801_data.feather")
    df = df.loc[~(df['e_type']==5)]

    return df, E_TYPE_DICT


def load_analyses():

    st.write('# Carbon Emission Analysis')
    st.write('## Data Analysis')
    st.write('### Insights of the Data')

    df, E_TYPE_DICT = load_resources()


    st.write('### 1). Time series of carbon emissions for various energy types')

    col1,  col2 = st.columns([1,1])

    with col1:

        e_type_scatter_charts = []
        for e_type in df['e_type'].unique():
            chart = go.Scatter(
                name=f"{E_TYPE_DICT[e_type]} Emission Levels",
                x=df.loc[~(df['Country']=='World')&(df['e_type']==e_type)]
                    .groupby('Year',as_index=False)
                    .agg({'CO2_emission':'sum'})
                    .loc[:,'Year'],
                y=df.loc[~(df['Country']=='World')&(df['e_type']==e_type)]
                    .groupby('Year',as_index=False)
                    .agg({'CO2_emission':'sum'})
                    .loc[:,'CO2_emission'],
            )

            e_type_scatter_charts.append(chart)

        fig = go.Figure()

        fig.add_traces(
            #Plot for carbon emission against energy types series
            e_type_scatter_charts
        )
        fig.update_layout(
            width=400,height=400,
            margin=dict(
                    t=0,l=0,r=10),
            # title="sheesh",
            legend=dict(
                    orientation="h"
            )
        )

        st.plotly_chart(fig,use_container_width=True)

    with col2:
        st.markdown("""
        We plotted the CO2 emissions categorized by their energy types.
        The curve appears to be heavily influenced by **coal sources** of emissions. Natural gases are also causative, though the gradient is not as severe. Renewable energies & nuclear were recorded to not have any carbon emissions. This is also seen in the chart below

        """)

        fig = go.Figure(
            data = [
            go.Pie(
                labels=[E_TYPE_DICT.get(i) for i in
                        list(df.loc[~(df["Country"]=="World")]
                            .groupby("e_type", as_index=False)
                            .agg({"CO2_emission":"mean"})
                            .loc[:,"e_type"].unique())],
                values=df.loc[~(df["Country"]=="World")]
                            .groupby("e_type", as_index=False)
                            .agg({"CO2_emission":"mean"})
                            .loc[:,"CO2_emission"]
            )

            ]
        )

        fig.update_layout(
            width=600,height=300,
            margin=dict(
                t=0,l=0,r=2,
            ),
            legend=dict(
                orientation="h"
            ),
            title={
                # "text":"Cumulative emissions for the time period",
                "y":0.02
                }
        )


        st.plotly_chart(fig, use_container_width=True)

    st.write('### 2). Timeseries plot of Carbon emission and GDP')

    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown(
        """There has been a steady constant increase in carbon emissions & GDP worldwide since the start of the century. The data shows a gradual reduction towards the end(2016), showing significant changes in human processes that cause emissions.
        A keen note on the chart, is the bleep that occured  between 2008 - 2010, which may have been caused by economic slowdown during the global financial crisis. It is evident that the emissions lagged against the GDP data during the period.
        """
        )    
    with col2:
        
        fig = make_subplots(specs=[[{'secondary_y':True}]])
        #plot figure for carbon & gdp
        print(df.columns)
        fig.add_trace(
            go.Scatter(
                name="Carbon Emission",
                x=df.loc[(df['Country']=='World')]
                    .groupby('Year', as_index=False)
                    .agg({'CO2_emission':'sum'})
                    .loc[:,"Year"],
                y=df.loc[(df['Country']=='World')]
                    .groupby('Year', as_index=False)
                    .agg({'CO2_emission':'sum',
                        'GDP':'sum'})
                    .loc[:,"CO2_emission"],
                
            ),secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                name="GDP",
                x=df.loc[df['Country']=='World']
                    .groupby('Year', as_index=False)
                    .agg({'GDP':"sum",})
                    .loc[:,"Year"],
                y=df.loc[df['Country']=='World']
                    .groupby('Year', as_index=False)
                    .agg({'GDP':"sum",})
                    .loc[:,"GDP"],
                mode="lines+markers"
            )
        )

        fig.update_layout(
            width=800,
            margin=dict(
                t=0, r=0, l=0),
            legend=dict(
                orientation="h", bgcolor='rgba(0,0,0,0)',
                yanchor='top',y=0.95,)
        )
        fig.update_yaxes(secondary_y=True, showgrid=False, title="GDP")
        fig.update_yaxes(secondary_y=False, title="Carbon Emissions")
        fig.update_xaxes(title="Year")

        st.plotly_chart(fig, use_container_width=True)


    
    st.write('### 2). Greatest emitters in the dataset')

    col1,  col2 = st.columns([1,2])

    df, E_TYPE_DICT = load_resources()


    with col2:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=df.loc[~(df['Country']=='World')]
                    .groupby('Country', as_index=False)
                    .agg({'CO2_emission':'sum'})
                    .sort_values('CO2_emission', ascending=False)
                    .loc[:,'Country'][:20],
                y=df.loc[~(df['Country']=='World')]
                    .groupby('Country', as_index=False)
                    .agg({'CO2_emission':'sum'})
                    .sort_values('CO2_emission', ascending=False)
                    .loc[:,'CO2_emission'][:20],
            )
        )

        fig.update_layout(
            margin=dict(t=0,l=0),
        )
        fig.update_xaxes(tickangle=45)

        st.plotly_chart(fig, use_container_width=True)

    with col1:
            st.markdown("""
            Some of the causes of heightened carbon emission are majorly caused by increased human activities, in the industries of infrastructure, manufacturing & societal development. A possible hypothesis would be: are these activities in any way linked to the population of the state? Does a higher population correlate to the emissions experienced?
            In the plot we have the top CO2 emitters in the dataset. Unsurprisingly, these are the major economies of the world.
            """)


    col1, col2 = st.columns([1,1])
