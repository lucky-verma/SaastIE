import streamlit as st
import os
import numpy as np
import pandas as pd
import argparse
import torch
from PIL import Image

from donut import DonutModel

# Streamlit configs
st.set_page_config(page_title='PPG2ABP',
                   page_icon=':heartpulse:',
                   layout='wide',
                   initial_sidebar_state='auto')

hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set the page title
st.title("SaastIE")



