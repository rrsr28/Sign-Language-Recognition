import cv2
import time
import math

import streamlit as st

# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide
st.set_page_config(page_title="Sign Language Recognizer", page_icon="👐", layout="wide")
st.markdown("""<h1 style='text-align: center;'>Sign Language Recognizer</h1><hr><br>""", unsafe_allow_html=True)

st.header("Welcome to Sign Language Recognizer!")
st.markdown("""
    This application helps you collect a dataset of hand signs, train a machine learning model, 
    and predict sign language gestures.
    To get started, navigate to the "Dataset Collector" page to capture hand sign images for your dataset.
    After collecting a sufficient number of images, move on to the "Model Training" page to train a model.
    Once the model is trained, you can use the "Sign Prediction" page to predict hand signs. """)