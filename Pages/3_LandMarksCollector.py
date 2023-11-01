import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from sklearn.model_selection import GridSearchCV

from Models import SVMModel
from Models.SVMModel import SVMModelc

# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide
st.set_page_config(page_title="Sign Language Recognizer", page_icon="👐", layout="wide")

st.markdown("""<h1 style='text-align: center;'>Sign Language CSV Dataset Collector</h1><hr><br>""", unsafe_allow_html=True)
st.subheader("Get the landmark points and their components for your dataset.") 

columns = []
for i in range(1, 22):
    columns.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
columns.append('label')
print(columns)

def image_processed(file_path):
    
    hand_img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

    output = hands.process(img_flip)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]

        data = str(data)

        data = data.strip().split('\n')

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
        return(np.zeros([1,63], dtype=int)[0])

def make_csv():
    
    mypath = 'Datasets/Images'
    file_name = open('Dataset.csv', 'w')

    for each_folder in os.listdir(mypath):
        if '._' in each_folder:
            pass

        else:
            for each_number in os.listdir(mypath + '/' + each_folder):
                if '._' in each_number:
                    pass
                
                else:
                    label = each_folder

                    file_loc = mypath + '/' + each_folder + '/' + each_number

                    data = image_processed(file_loc)
                    try:
                        for id,i in enumerate(data):
                            if id == 0:
                                print(i)
                            
                            file_name.write(str(i))
                            file_name.write(',')

                        file_name.write(label)
                        file_name.write('\n')
                    
                    except:
                        file_name.write('0')
                        file_name.write(',')

                        file_name.write('None')
                        file_name.write('\n')
       
    file_name.close()
    st.success('CSV Created Successfully !')

def build_model_svc():
    svm = SVMModelc('Dataset.csv')
    svm.save_model('Models/SVMModel.pkl')

make_csv()
df = pd.read_csv('Dataset.csv', header=None, names=columns)
st.dataframe(df)

#build_model_svc()
# {'C': 10, 'degree': 2, 'gamma': 1, 'kernel': 'poly'}
# 0.9835897435897436