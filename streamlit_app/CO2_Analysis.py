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


if __name__ == '__main__':
    main()