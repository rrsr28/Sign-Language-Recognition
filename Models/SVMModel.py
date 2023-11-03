import os
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

class SVMModelc:

    def bestModel(self):
        columns = []
        for i in range(1, 22):
            columns.extend([f'landmark_{i}x', f'landmark{i}y', f'landmark{i}_z'])
        columns.append('label')

        data = pd.read_csv(self.dataset_file)
        data.columns = columns

        X = data.iloc[:, :-1]
        Y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        """
        param_grid = { 
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': [0.01, 0.1, 1],
            'degree': [2, 3, 4]
        }
        """

        #grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
        grid_search = SVC(C=10, gamma=1, kernel='poly') # The best model
        grid_search.fit(X_train, y_train)

        #best_params = grid_search.best_params_

        #best_svm = SVC(**best_params)
        #best_svm.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        #print(best_params)
        print(accuracy)

        return grid_search

    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
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

        self.best_svm = self.bestModel()
        self.best_svm.fit(X_train, y_train)

        y_pred = self.best_svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')

        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("Recall Score:", recall)
        print("Precision Score:", precision)

    def save_model(self, model_filename):
        pickle.dump(self.best_svm, open(model_filename, 'wb'))

    def load_model(self, model_filename):
        self.best_svm = pickle.load(open(model_filename, 'rb'))