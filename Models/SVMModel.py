import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score

columns = []
for i in range(1, 22):
    columns.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
columns.append('label')

data = pd.read_csv('Dataset.csv')
data.columns = columns

# Add "No Hands Detected" for rows that are full of zeroes (excluding the label column)
data.loc[data.iloc[:, :-1].eq(0).all(axis=1), 'label'] = 'No Hands Detected'

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

print("Features shape = ", X.shape)
print("Labels shape = ", Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
svm = SVC(C=10, gamma=0.1, kernel='rbf')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print(y_pred)
print("\n\n")
print("Accuracy score = ", accuracy_score(y_test, y_pred))
print("F1 score = ", f1_score(y_test, y_pred, average='weighted'))
print("Recall score = ", recall_score(y_test, y_pred, average='weighted'))
print("Precision score = ", precision_score(y_test, y_pred, average='weighted'))
print("\n\n")
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

pickle.dump(svm, open('Models/SVMModel.pkl', 'wb'))