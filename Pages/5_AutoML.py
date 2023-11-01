import os
import pandas as pd
import autokeras as ak
import streamlit as st
from sklearn.model_selection import train_test_split

# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide
st.set_page_config(page_title="Sign Language Recognizer", page_icon="👐", layout="wide")

st.markdown("""<h1 style='text-align: center;'>Sign Language CSV Dataset Collector</h1><hr><br>""", unsafe_allow_html=True)
st.subheader("")

current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_directory, '..', 'Dataset.csv')

data = pd.read_csv(csv_file_path)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = ak.StructuredDataClassifier(max_trials=10)
clf.fit(X_train, y_train, epochs=10)

score = clf.evaluate(X_test, y_test)
print(f"Test loss: {score[0]}, Test accuracy: {score[1]}")

predictions = clf.predict(X_test)

print("Best AutoKeras Model Summary:")
clf.tuner.get_best_model().summary()