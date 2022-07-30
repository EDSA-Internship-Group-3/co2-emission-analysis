import sys
from PIL import Image

# Streamlit Dependencies
import streamlit as st


# Data handling dependencies
# import pandas as pd
# import numpy as np

# Custom Libraries
from load_xgboost import xgboost_model
#from recommenders.content_based import content_model

st.set_page_config(
    layout='wide'
)


# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Introduction","Data Analysis", "Model Deployment", "Conclusion"]

    # -------------------------------------------------------------------
    page_selection = st.sidebar.radio("Page Navigation", page_options)

    
    # -------------------------------------------------------------------
    #        INTRODUCTION PAGE
    # -------------------------------------------------------------------

    if page_selection == "Introduction":
        # Header contents
        col1, col2 = st.columns([1,1])

        with col1:
            st.write('# CO2 Emission Analysis')
            st.write('## Ai Glass Data Science Team')

        col1, col2, col3 = st.columns([1,1,1])

        st.markdown(
            """
            <h2 style='text-align: center;'>
                Meet the Team
            </h2>
            """,
            unsafe_allow_html=True
        )

        cols = col1, col2, col3, \
        col5, col5, col6 = st.columns([1,1,1,1,1,1])

        team_members = ["Teddy Waweru", "Ahmad Barde","Raphael Mbonu",
            "Ebere Ezeudemba","Dare Nuges","Israel Ezema"]

        for col,team_member in zip(cols, team_members):
            with col:
                img = Image.open(f'imgs/{team_member}.jpeg')
                st.image(img,use_column_width=True,
                    caption = team_member,width=1000)


        st.write('## Problem Statement')

        st.write('## Overview of Our Solution')



    # -------------------------------------------------------------------
    #        Data analysis PAGE
    # -------------------------------------------------------------------
    
    if page_selection == "Data Analysis":
        # Header contents
        st.write('# CO2 Emission Analysis')
        st.write('## Data Analysis')
        st.write('### Insights of the Data')
        # st.image('resources/imgs/Image_header.png',use_column_width=True)

    # -------------------------------------------------------------------
    #        MODEL DEPLOYMENT PAGE
    # -------------------------------------------------------------------
    
    if page_selection == "Model Deployment":
        # Header contents
        st.write('# CO2 Emission Analysis')
        st.write('## Model Deployment')
        # st.image('resources/imgs/Image_header.png',use_column_width=True)
        if st.button('Refresh Page'):
            xgboost_model()

        xgboost_model()


    
    # -------------------------------------------------------------------
    #        CONCLUSION PAGE
    # -------------------------------------------------------------------
    
    if page_selection == "Conclusion":
        # Header contents
        st.write('# CO2 Emission Analysis')
        st.write('## Conclusion')
        # st.image('resources/imgs/Image_header.png',use_column_width=True)
    


if __name__ == '__main__':
    main()