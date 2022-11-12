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
st.title("SaastIE")

# Set the page subtitle
st.markdown("## A OCR-free transformer based model for Document Understanding")

# Set the page description
st.markdown("### SaastIE is a transformer based model for document understanding. It can be used to extract information from documents such as invoices, receipts, contracts, etc. It can also be used to extract information from forms such as application forms, feedback forms, etc.")

# Set the API to be used
API_ENDPOINT = "http://127.0.0.1:6969/"

# Select the tabs for document understanding task
clasification, pasrsing, vqa = st.tabs(["Classification", "Parsing", "Visual Question Answering"])

# Classification tab
if clasification:
    # Set the page title
    st.subheader("Document Classification")

    # Upload the document
    uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'png', 'jpeg'])

    # If the file is uploaded
    if uploaded_file is not None:
        # Read the file
        image = Image.open(uploaded_file)

        # # Convert the file to numpy array
        # image = np.array(image)

        # Send the file to the API
        response = requests.post(API_ENDPOINT + "classifier", files={'file': uploaded_file})

        # Get the response from the API
        response = response.json()

        # Get the predicted class
        predicted_class = response['predicted_class']

        # Get the predicted class probability
        st.success(f"Predicted class: {predicted_class}")

        st.snow()