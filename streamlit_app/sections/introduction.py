# -------------------------------------------------------------------
#        DEFINES THE INTRODUCTION SECTION
# -------------------------------------------------------------------

from PIL import Image
import streamlit as st
from .load_markdown import load_markdown_file



def load_introduction():

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


    st.write('## Introduction')
    st.markdown(load_markdown_file('resources/markdowns/introduction.md'))

    st.write('### Overview of Our Solution')
    st.markdown(load_markdown_file('resources/markdowns/solution_overview.md'))
    
    st.write('### Problem Statement')
    st.markdown(load_markdown_file('resources/markdowns/problem_statement.md'))


