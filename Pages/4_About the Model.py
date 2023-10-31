import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score

# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide
st.set_page_config(page_title="Sign Language Recognizer", page_icon="👐", layout="wide")

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

st.markdown("""<h1 style='text-align: center;'>About the Model</h1><br>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Model Details', 'Dataset', 'Performance', 'Classification Report', 'Summary of Metrics'])
with tab1:
    st.write("SVM Kernel: Radial Basis Function (RBF)")
    st.write("C: 10")
    st.write("Gamma: 0.1")

with tab2:
    st.write("The model was trained on a dataset of hand gesture images.")

with tab3:
    st.write("The model achieved the following performance metrics:")
    st.write(f"- Accuracy Score: {accuracy_score(y, y_pred):.2f}")
    st.write(f"- F1 Score: {f1_score(y, y_pred, average='weighted'):.2f}")
    st.write(f"- Recall Score: {recall_score(y, y_pred, average='weighted'):.2f}")
    st.write(f"- Precision Score: {precision_score(y, y_pred, average='weighted'):.2f}")

with tab4:
    st.write("You can view the detailed classification report below:")
    st.code(classification_report(y, y_pred))