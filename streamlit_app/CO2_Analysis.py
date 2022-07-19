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
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        # st.image('resources/imgs/Image_header.png',use_column_width=True)



    # -------------------------------------------------------------------
    #        Data analysis PAGE
    # -------------------------------------------------------------------
    
    if page_selection == "Data Analysis":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        # st.image('resources/imgs/Image_header.png',use_column_width=True)

    # -------------------------------------------------------------------
    #        MODEL DEPLOYMENT PAGE
    # -------------------------------------------------------------------
    
    if page_selection == "Model Deployment":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        # st.image('resources/imgs/Image_header.png',use_column_width=True)

    
    # -------------------------------------------------------------------
    #        CONCLUSION PAGE
    # -------------------------------------------------------------------
    
    if page_selection == "Conclusion":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        # st.image('resources/imgs/Image_header.png',use_column_width=True)
    


if __name__ == '__main__':
    main()