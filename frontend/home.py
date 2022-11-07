import streamlit as st
import os
import numpy as np
import pandas as pd
import argparse
import torch
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
st.title("SaastIE")

# Set the page subtitle
st.markdown("## A OCR-free transformer based model for Document Understanding")

# Set the page description
st.markdown("### SaastIE is a transformer based model for document understanding. It can be used to extract information from documents such as invoices, receipts, contracts, etc. It can also be used to extract information from forms such as application forms, feedback forms, etc.")


