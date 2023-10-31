import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score

# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide
st.set_page_config(page_title="Sign Language Recognizer", page_icon="👐", layout="wide")

# Load the trained SVM model
with open('Models/SVMModel.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Define the columns and read the dataset
columns = []
for i in range(1, 22):
    columns.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
columns.append('label')

data = pd.read_csv('Dataset.csv')
data.columns = columns
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
y_pred = svm_model.predict(X)

st.title("About the Model")

st.write("This model was trained to recognize sign language gestures.")
st.write("It uses a Support Vector Machine (SVM) classifier to make predictions based on hand landmarks.")

st.subheader("Model Details")
st.write("SVM Kernel: Radial Basis Function (RBF)")
st.write("C: 10")
st.write("Gamma: 0.1")

st.subheader("Dataset")
st.write("The model was trained on a dataset of hand gesture images.")

st.subheader("Performance")
st.write("The model achieved the following performance metrics:")
st.write(f"- Accuracy Score: {accuracy_score(y, y_pred):.2f}")
st.write(f"- F1 Score: {f1_score(y, y_pred, average='weighted'):.2f}")
st.write(f"- Recall Score: {recall_score(y, y_pred, average='weighted'):.2f}")
st.write(f"- Precision Score: {precision_score(y, y_pred, average='weighted'):.2f}")

st.subheader("Classification Report")
st.write("You can view the detailed classification report below:")
st.code(classification_report(y, y_pred))

st.subheader("Summary of Metrics")
st.write(f"F1 Score: {f1_score(y, y_pred, average='weighted'):.2f}")
st.write(f"Recall Score: {recall_score(y, y_pred, average='weighted'):.2f}")
st.write(f"Precision Score: {precision_score(y, y_pred, average='weighted'):.2f}")

st.write("Thank you for using Sign Language Recognizer.")
