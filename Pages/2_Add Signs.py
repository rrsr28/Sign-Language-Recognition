import os
import cv2
import math
import time
import numpy as np
import streamlit as st
import mediapipe as mp

# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide
st.set_page_config(page_title="Sign Language Recognizer", page_icon="👐", layout="wide")

counter = 0

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

st.markdown("""<h1 style='text-align: center;'>Sign Language Image Dataset Collector</h1><hr><br>""", unsafe_allow_html=True)
st.subheader("Capture hand sign images for your Train dataset.")

folder = "Datasets/Images/"
folder += st.text_input("Enter the label for the dataset:", key="folder_input")

if folder and st.button("Capture Dataset", key="capture_button"):

    counter = 0
    create_folder(folder)

    cap = cv2.VideoCapture(0)
    while True:

        ret, frame = cap.read()
                
        if cv2.waitKey(1) & 0xFF == ord("s"):
            counter += 1
            print(counter)
            if counter <= 75:
                timestamp = time.time()
                cv2.imwrite(f'{folder}/Image_{timestamp}.jpg', frame)
            elif counter == 76:
                st.success("Dataset Collection Complete")
                if st.button("Start New Dataset", key="new_dataset_button"):
                    pass
            else:
                pass

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()