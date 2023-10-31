import os
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

class SVMModelc:
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.dataset_file = os.path.abspath(self.dataset_file)
        self.svmodel()

    def svmodel(self):
        columns = []
        for i in range(1, 22):
            columns.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
        columns.append('label')

        self.data = pd.read_csv(self.dataset_file)
        self.data.columns = columns
        self.data.loc[self.data.iloc[:, :-1].eq(0).all(axis=1), 'label'] = 'No Hand Detected'

        X = self.data.iloc[:, :-1]
        Y = self.data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        self.svm = SVC(C=10, gamma=0.1, kernel='rbf', probability=True)  # Enable probability estimates
        self.svm.fit(X_train, y_train)

        y_pred = self.svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')

        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("Recall Score:", recall)
        print("Precision Score:", precision)

    def save_model(self, model_filename):
        pickle.dump(self.svm, open(model_filename, 'wb'))

    def load_model(self, model_filename):
        self.svm = pickle.load(open(model_filename, 'rb'))