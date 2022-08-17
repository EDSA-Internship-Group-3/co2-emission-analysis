# -------------------------------------------------------------------
#        DEFINES THE CONCLUSION SECTION
# -------------------------------------------------------------------

import streamlit as st

from .load_markdown import load_markdown_file

def load_conclusion():
    # Header contents
    st.write('# CO2 Emission Analysis')
    st.write('## Conclusion')
    st.markdown(load_markdown_file('resources/markdowns/conclusion.md'))
