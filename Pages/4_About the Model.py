import pickle
import pandas as pd
import numpy as np
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score
from sklearn.svm import SVC  # Import the SVC class

# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide
st.set_page_config(page_title="Sign Language Recognizer", page_icon="👐", layout="wide")

# Load your trained SVM model
with open('Models/SVMModel.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

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

st.markdown("""<h1 style='text-align: center;'>About the Model</h1><br>""", unsafe_allow_html=True)

tabs = st.tabs(["Model Details", "Dataset", "Performance", "LIME Explanation", "Classification Report"])
with tabs[0]:
    st.write("This model was trained to recognize sign language gestures.")
    st.write("It uses a Support Vector Machine (SVM) classifier to make predictions based on hand landmarks.")
    st.write("SVM Kernel: Radial Basis Function (RBF)")
    st.write("C: 10")
    st.write("Gamma: 0.1")

with tabs[1]:
    st.write("The model was trained on a dataset of hand gesture images.")

with tabs[2]:
    st.write("The model achieved the following performance metrics:")
    st.write(f"- Accuracy Score: {accuracy_score(y, y_pred):.2f}")
    st.write(f"- F1 Score: {f1_score(y, y_pred, average='weighted'):.2f}")
    st.write(f"- Recall Score: {recall_score(y, y_pred, average='weighted'):.2f}")
    st.write(f"- Precision Score: {precision_score(y, y_pred, average='weighted'):.2f}")

with tabs[3]:
    st.write("You can view the LIME explanation for a specific instance below:")
    # Select a specific instance for explanation
    instance_idx = 2  # Replace with the index of the instance you want to explain
    instance = X.iloc[instance_idx].values

    feature_names = X.columns
    explainer = LimeTabularExplainer(X.values, mode="classification", feature_names=feature_names)
    explanation = explainer.explain_instance(instance, svm_model.predict_proba)
    explanation_html = explanation.as_html()

    explanation_image = explanation.as_pyplot_figure()
    st.pyplot(explanation_image)
    st.components.v1.html(explanation_html, height=500)

with tabs[4]:
    st.write("You can view the detailed classification report below:")
    st.code(classification_report(y, y_pred))
