import os
import cv2
import math
import time
import numpy as np
import streamlit as st
import mediapipe as mp

# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide
st.set_page_config(page_title="Sign Language Recognizer", page_icon="👐", layout="wide")

offset = 30
imgSize = 400
counter = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

def bounding_box(hand_landmarks):
    x_min = min([landmark.x for landmark in hand_landmarks.landmark])
    x_max = max([landmark.x for landmark in hand_landmarks.landmark])
    y_min = min([landmark.y for landmark in hand_landmarks.landmark])
    y_max = max([landmark.y for landmark in hand_landmarks.landmark])
    return int(x_min * frame.shape[1]), int(y_min * frame.shape[0]), int((x_max - x_min) * frame.shape[1]), int((y_max - y_min) * frame.shape[0])

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

st.title("Sign Language Dataset Collector")
st.subheader("Capture hand sign images for your Train dataset.")

folder = "Dataset/Train/"
folder += st.text_input("Enter the label for the dataset:", key="folder_input")

if folder and st.button("Capture Dataset", key="capture_button"):

    counter = 0
    create_folder(folder)

    cap = cv2.VideoCapture(0)
    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            x, y, w, h = bounding_box(hand_landmarks)

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            try:
                imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]

                imgCropShape = imgCrop.shape

                aspect_ratio = h / w

                if aspect_ratio > 1:
                    k = imgSize / h
                    w_cal = math.ceil(k * w)
                    img_resize = cv2.resize(imgCrop, (w_cal, imgSize))
                    w_gap = math.ceil((imgSize - w_cal) / 2)
                    imgWhite[:, w_gap:w_cal + w_gap] = img_resize

                else:
                    k = imgSize / w
                    h_cal = math.ceil(k * h)
                    img_resize = cv2.resize(imgCrop, (imgSize, h_cal))
                    h_gap = math.ceil((imgSize - h_cal) / 2)
                    imgWhite[h_gap:h_cal + h_gap, :] = img_resize

                cv2.imshow("Captured Image", imgWhite)
                
                if cv2.waitKey(1) & 0xFF == ord("s"):
                    counter += 1
                    print(counter)
                    if counter <= 100:
                        timestamp = time.time()
                        cv2.imwrite(f'{folder}/Image_{timestamp}.jpg', imgWhite)
                    elif counter == 101:
                        st.success("Dataset Collection Complete")
                        if st.button("Start New Dataset", key="new_dataset_button"):
                            pass
                    else:
                        pass

            except cv2.error as e:
                print(f"Error during image processing.\nThe ROI was too close to the edge of the frame.\n")

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()