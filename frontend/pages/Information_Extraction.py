import streamlit as st
import requests
import io
from PIL import Image

# Streamlit configs
st.set_page_config(page_title='SaasTIE',
                   page_icon=':sunglasses:',
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
st.title("SaasTIE")

# Set the API to be used
API_ENDPOINT = "http://127.0.0.1:6969/"

# Set the page title
st.subheader("Information Extraction")

# Upload the document
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['jpg', 'png', 'jpeg'])

# If the file is uploaded
if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=False, width=500)

        # Read the file
        image = Image.open(uploaded_file)

        # Convert the iamge to BytesIO
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
    
    # button to activate the API
    button = st.button("Extract")

    if button:
        with col2:
            
            # Send the file to the API
            response = requests.post(API_ENDPOINT + "parser", files={'file': img_byte_arr})

            # Get the response from the API
            response = response.json()

            # Get the predicted class probability
            st.json(response, expanded=False)

            st.snow()