import os
import pandas as pd
import autokeras as ak
import streamlit as st
from sklearn.model_selection import train_test_split

# Set Streamlit theme to dark, enable "Run on Save," and set the default layout to wide
st.set_page_config(page_title="Sign Language Recognizer", page_icon="👐", layout="wide")

# Set title and description
st.markdown("""<h1 style='text-align: center;'>AutoML</h1><hr><br>""", unsafe_allow_html=True)
st.subheader("AutoML using AutoKeras")

current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_directory, '..', 'Dataset.csv')

data = pd.read_csv(csv_file_path)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = ak.StructuredDataClassifier(max_trials=10)

# Fit the model and get the training history
history = clf.fit(X_train, y_train, epochs=10)

# Display the training history
st.subheader("Training History")
st.line_chart(history.history['loss'])

# Get the best AutoKeras model summary
best_model = clf.tuner.get_best_model()

# Display the best model summary
st.subheader("Best AutoKeras Model Summary")
best_model.summary()

# Evaluate the model
score = clf.evaluate(X_test, y_test)

# Display the number of epochs and test results
st.subheader("Training Information")
st.write("Number of Epochs: 10")
st.write("Test Loss:", score[0])
st.write("Test Accuracy:", score[1])

predictions = clf.predict(X_test)