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
    and predict sign language gestures.<br><hr>""", unsafe_allow_html=True)

st.header("How to use")
st.write("""
1. **Home Page**:
   - When you first open the website, you will see the home page with a brief introduction and instructions.
   - To get started, you can navigate to different pages using the links mentioned on the home page.

2. **Prediction Page**:
   - In this section, the web application uses a trained Support Vector Machine (SVM) model to predict hand signs based on hand landmarks detected using the MediaPipe library.
   - The webcam feed will display on the page, and your hand signs will be recognized in real-time.
   - The predicted sign will be displayed on the page as you perform the signs. Please note that the SVM model is loaded from the "SVMModel.pkl" file.

3. **Adding/Increasing Dataset Page**:
   - This page allows you to capture hand sign images for your training dataset.
   - You need to provide a label for the dataset, and then you can click the "Capture Dataset" button to start capturing images.
   - The images are saved in a folder named after the provided label.
   - You can capture up to 75 images, and a success message will appear when the collection is complete. You can then start a new dataset.

4. **CSV/Landmark Generator and Model Generator Page**:
   - After collecting images, you can go to this page to generate a CSV file containing landmark data from the collected images.
   - The `make_csv()` function reads the images, processes them to extract landmarks using MediaPipe, and saves the landmark data along with labels to a CSV file.
   - The `build_model_svc()` function uses the generated CSV data to train an SVM model with specific hyperparameters (C, degree, gamma, kernel).
   - To know more about more the landmarks refer https://media.geeksforgeeks.org/wp-content/uploads/20210802154942/HandLandmarks.png

5. **About the Model Page**:
   - This page provides information about the SVM model used for sign language gesture recognition.
   - It includes details about the SVM kernel (polynomial), hyperparameters (C, gamma), and information about the dataset.
   - Displays performance metrics for the SVM model, such as accuracy, F1 score, recall, and precision.
   - It also provides a LIME (Local Interpretable Model-Agnostic Explanations) explanation for a specific instance from the dataset. You can view the feature importance for that instance.
""")