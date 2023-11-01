import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import threading

# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide
st.set_page_config(page_title="Sign Language Recognizer", page_icon="👐", layout="wide")

detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def capture_image(img):
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        try:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            return img, imgWhite

        except cv2.error as e:
            print(f"Error during image processing.\nThe ROI was too close to the edge of the frame.\n")
            return img, None

    return img, None

def main():
    global counter
    st.title("Sign Language Dataset Collector")
    st.subheader("Capture hand sign images for your Train dataset.")

    folder_name = "Dataset/Train/"
    folder_name += st.text_input("Enter the label for the dataset:", key="folder_input")

    if folder_name and st.button("Capture Dataset", key="capture_button"):
        counter = 0
        create_folder(folder_name)

        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
            img, img_white = capture_image(img)

            if img_white is not None:
                cv2.imshow("Captured Image", img_white)

                if cv2.waitKey(1) & 0xFF == ord("s"):
                    counter += 1
                    print(counter)
                    if counter <= 100:
                        timestamp = time.time()
                        cv2.imwrite(f'{folder_name}/Image_{timestamp}.jpg', img_white)
                    elif counter == 101:
                        st.success("Dataset Collection Complete")
                        if st.button("Start New Dataset", key="new_dataset_button"):
                            pass
                    else:
                        pass

            cv2.imshow("Video Stream", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

main()