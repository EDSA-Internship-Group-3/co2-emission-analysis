# Streamlit Dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np



# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Introduction","Data Analysis", "Model Deployment", "Conclusion"]

    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)

    
    # -------------------------------------------------------------------
    #        INTRODUCTION PAGE
    # -------------------------------------------------------------------

    if page_selection == "Introduction":
        # Header contents
        st.write('# CO2 Emission Analysis')
        st.write('## Ai Glass Data Science Team')
        st.write('### Meet The Team')
        st.image('streamlit_app/imgs/teddy.jpeg',use_column_width=True, caption = 'Teddy')
        st.image('streamlit_app/imgs/ahmad.jpeg',use_column_width=True, caption = 'Ahmad')
        st.image('streamlit_app/imgs/raph.jpeg',use_column_width=True, caption = 'Raphael')
        st.image('streamlit_app/imgs/ebere.jpeg',use_column_width=True, caption = 'Ebere')
        st.image('streamlit_app/imgs/dare.jpeg',use_column_width=True, caption = 'Dare')
        st.image('streamlit_app/imgs/israel.png',use_column_width=True, caption = 'Israel')

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