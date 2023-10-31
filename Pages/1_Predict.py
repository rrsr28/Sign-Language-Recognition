import cv2
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from threading import Thread

with open('Models/SVMModel.pkl', 'rb') as f:
    svm = pickle.load(f)

def image_processed(hand_img):
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    
    output = hands.process(img_flip)
    hands.close()
    
    try:
        data = output.multi_hand_landmarks[0]
        data = str(data).strip().split('\n')
        
        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        
        return(clean)

    except:
        return np.zeros([1, 63], dtype=int)[0]

st.set_page_config(page_title="Sign Language Recognizer", page_icon="👐", layout="wide")

st.markdown("""<h1 style='text-align: center;'>Sign Language Recognition</h1><hr><br>""", unsafe_allow_html=True)
st.subheader("Predict hand signs using the trained model.")

st.write("""
This app uses a trained Support Vector Machine (SVM) model to predict hand signs based on hand landmarks detected using the MediaPipe library.""")

st.write("""
- The model is trained on a dataset of hand signs, capturing the 3D coordinates (x, y, z) of 21 hand landmarks.
- Hand signs are recognized in real-time using the webcam feed.""")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Cannot open camera. Please check your camera connection.")
    st.stop()

while True:
    ret, frame = cap.read()

    if not ret:
        st.error("Can't receive frame (stream end?). Exiting ...")
        break

    data = image_processed(frame)
    data_2d = np.array(data).reshape(1, -1)
    columns = [f'landmark_{i}_{coord}' for i in range(1, 22) for coord in ['x', 'y', 'z']]
    data_df = pd.DataFrame(data_2d, columns=columns)

    y_pred = svm.predict(data_df)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 100)
    font_scale = 3
    color = (255, 0, 0)
    thickness = 5
    frame = cv2.putText(frame, str(y_pred[0]), org, font, font_scale, color, thickness, cv2.LINE_AA)

    if len(data) > 0:
            hand_bbox = cv2.boundingRect(np.array(data_2d[:, :2], dtype=int))
            x, y, w, h = hand_bbox
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()