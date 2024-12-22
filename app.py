import json 

# import keras
import requests
import streamlit as st 
# from streamlit_lottie import st_lottie # Import necessary libraries
from tensorflow.keras.models import load_model

# Load the model globally
# model = keras.models.load_model('sequential_saved_model.h5')
import tensorflow as tf





# model = load_model('sequential_saved_model.h5')



st.set_page_config(page_title="Real Time Sentiment Analysis", page_icon="🤖", layout="wide")

# Use Streamlit's caching to load the model only once
@st.cache_resource
def load_model_once():
    # Replace with the correct path to your saved model file
    return load_model('sequential_saved_model.h5')

# Load the model
model = load_model_once()
 
    
# def load_lottieurl(url:str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# lottie_anim = load_lottieurl("https://lottie.host/e066955b-f983-46ad-9e40-caa3b8c99288/fG1IT9DrVy.json")



# st_lottie(lottie_anim,speed= 2 , height=350, width=350, key=None)

# Page setup

home_page = st.Page(
    page="views/home.py",
    title="Real Time Sentiment Analysis",
    default=True,
)
Stored_Data = st.Page(
    page="views/store.py",
    title="Upload Data",
    icon="📷",
    # args=(model,)  # Pass the model to store.py
    
)

Live_Data = st.Page(
    page="views/live.py",
    title="Capture",
    icon="📹",
    
)


# NAVIGATION #

pg = st.navigation(
    {
        "Info":[home_page],
        "Section": [Stored_Data, Live_Data],
    }
)




# st.sidebar.text("Made by Team :)")

# RUN #
pg.run()
