from PIL import Image
from pathlib import Path

# Streamlit Dependencies
import streamlit as st


# Custom Libraries
from sections.introduction import load_introduction
from sections.model_deployment import xgboost_model
from sections.data_analysis import load_analyses
from sections.conclusion import load_conclusion

st.set_page_config(
    layout='wide'
)

# App declaration
def main():

    page_options = ["Introduction","Data Analysis", "Model Deployment", "Conclusion"]

    # -------------------------------------------------------------------
    # Selector for sections of the application.
    page_selection = st.sidebar.radio("Page Navigation", page_options)


    if page_selection == "Introduction":
        load_introduction()

    
    if page_selection == "Data Analysis":
        load_analyses()


    if page_selection == "Model Deployment":
        xgboost_model()

    
    if page_selection == "Conclusion":
        load_conclusion()
    


if __name__ == '__main__':
    main()