import io
import streamlit as st
import os
import numpy as np
import pandas as pd
import argparse
import torch
import requests

from PIL import Image

from donut import DonutModel

# Streamlit configs
st.set_page_config(page_title='SaasTIE', page_icon=':sunglasses:', layout='wide', initial_sidebar_state='auto')

hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set the page title
st.title("SaasTIE")

# Set the page subtitle
st.markdown("## A OCR-free transformer based model for Document Understanding")

# Set the page description
st.markdown("#### SaastIE is a transformer based model for document understanding. It can be used to extract information from documents such as invoices, receipts, contracts, etc. It can also be used to extract information from forms such as application forms, feedback forms, etc.")

# Create 3 columns
col1, col2, col3 = st.columns(3)

with col2:
    # show the task image
    st.image("src/1.png", caption='Information Extraction')

with col1:
    # show the task image
    st.image("src/2.png", caption='Classification')

with col3:
    # show the task image
    st.image("src/3.png", caption='Question Answering')

