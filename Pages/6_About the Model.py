import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.svm import SVC
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score

# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide
st.set_page_config(page_title="Sign Language Recognizer", page_icon="👐", layout="wide")


columns = []
for i in range(1, 22):
    columns.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
columns.append('label')

data = pd.read_csv('Dataset.csv')
data.columns = columns
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

svm_model = SVC(C=10, gamma=1, kernel='poly', probability=True)  # Enable probability estimates
svm_model.fit(X, y)
y_pred = svm_model.predict(X)

st.title("Sign Language Gesture Recognition")

tabs = st.tabs(["Model Details", "Dataset", "Performance", "LIME Explanation", "Classification Report"])

with tabs[0]:
    st.header("Model Details")
    st.write(
        "This model was trained to recognize sign language gestures using a Support Vector Machine (SVM) classifier.")
    st.write("SVM Kernel: Polynomial Kernel (poly)")
    st.write("C: 10")
    st.write("Gamma: 1")

with tabs[1]:
    st.header("Dataset Information")
    st.write("The model was trained on a dataset of hand gesture images.")

with tabs[2]:
    st.header("Performance Metrics")
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    precision = precision_score(y, y_pred, average='weighted')

    st.write(f"Accuracy Score: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    st.write(f"Recall Score: {recall:.2f}")
    st.write(f"Precision Score: {precision:.2f}")

with tabs[3]:
    st.header("LIME Explanation")
    st.write("You can view the LIME explanation for a specific instance below. This explanation shows the feature "
             "importance for a sample from the dataset.")

    instance_idx = 1  # Replace with the index of the instance you want to explain
    instance = X.iloc[instance_idx].values

    feature_names = X.columns
    explainer = LimeTabularExplainer(X.values, mode="classification", feature_names=feature_names)
    explanation = explainer.explain_instance(instance, svm_model.predict_proba)

    st.pyplot(explanation.as_pyplot_figure(), use_container_width=True)

    st.write("Feature Importance:")
    feature_weights = [(feature, weight) for feature, weight in explanation.as_list()]
    explanation_df = pd.DataFrame(feature_weights, columns=["Feature", "Weight"])
    st.bar_chart(explanation_df.set_index("Feature"))

with tabs[4]:
    st.header("Classification Report")
    st.write(
        "You can view the detailed classification report below. This report provides a comprehensive evaluation of the model's performance.")
    st.code(classification_report(y, y_pred))
