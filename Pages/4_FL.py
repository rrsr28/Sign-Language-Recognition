import os
import copy
import pickle
import pandas as pd
from sklearn.svm import SVC
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

columns = []
for i in range(1, 22):
    columns.extend([f'landmark_{i}x', f'landmark{i}y', f'landmark{i}_z'])
columns.append('label')

current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_directory, '..', 'Dataset.csv')

data = pd.read_csv(csv_file_path)
data.columns = columns
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X = X_train
Y = y_train

num_clients = 5
num_rounds = 5

global_model = SVC(C=10, gamma=0.1, kernel='rbf')

client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
X_clients = [X[i:i + len(X) // num_clients] for i in range(0, len(X), len(X) // num_clients)]
Y_clients = [Y[i:i + len(Y) // num_clients] for i in range(0, len(Y), len(Y) // num_clients)]

# Federated learning
for round in range(num_rounds):
    for i in range(num_clients):
        # Train the client model on client data
        client_models[i].fit(X_clients[i], Y_clients[i])

    # Update client models with the global model (simple averaging)
    for i in range(num_clients):
        client_models[i] = copy.deepcopy(global_model)

current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_directory, '..', 'Models/FL_Client ')

client_id = 0
for i in range(num_clients):
    client_id += 1
    client_models[i].fit(X_clients[i], Y_clients[i])
    model_filename = csv_file_path + str(client_id) + str(".pkl")
    with open(model_filename, 'wb') as model_file:
        pickle.dump(client_models[i], model_file)
    print(f"Local model for client {client_id} saved to {model_filename}")

client_predictions = []
for i in range(num_clients):
    client_prediction = client_models[i].predict(X_test)
    client_predictions.append(client_prediction)

def majority_voting(predictions):
    global_predictions = []
    for i in range(len(predictions[0])):
        votes = [predictions[j][i] for j in range(num_clients)]
        majority = Counter(votes).most_common(1)
        global_predictions.append(majority[0][0])
    return global_predictions

global_predictions = majority_voting(client_predictions)

accuracy = accuracy_score(y_test, global_predictions)
print("Accuracy:", accuracy)