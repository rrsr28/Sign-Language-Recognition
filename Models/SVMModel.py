import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

class SVMModel:
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.load_dataset()
        self.train_model()

    def load_dataset(self):
        columns = []
        for i in range(1, 22):
            columns.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
        columns.append('label')

        self.data = pd.read_csv(self.dataset_file)
        self.data.columns = columns
        self.data.loc[self.data.iloc[:, :-1].eq(0).all(axis=1), 'label'] = 'No Hand Detected'

    def train_model(self):
        X = self.data.iloc[:, :-1]
        Y = self.data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        self.svm = SVC(C=10, gamma=0.1, kernel='rbf')
        self.svm.fit(X_train, y_train)

    def evaluate_model(self):
        y_pred = self.svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')

        return accuracy, f1, recall, precision

    def save_model(self, model_filename):
        pickle.dump(self.svm, open(model_filename, 'wb'))

if __name__ == "__main__":
    # Replace 'Dataset.csv' with the actual dataset file path
    svm = SVMModel('Dataset.csv')
    accuracy, f1, recall, precision = svm.evaluate_model()
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Recall Score:", recall)
    print("Precision Score:", precision)
    svm.save_model('SVMModel.pkl')
