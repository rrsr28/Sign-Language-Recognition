import os
import h2o
import pandas as pd
import autokeras as ak
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split

current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_directory, '..', 'Dataset.csv')

data = pd.read_csv(csv_file_path)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize an H2O cluster
h2o.init()
data = h2o.H2OFrame(data)

x = data.columns[:-1]
y = data.columns[-1]

# Initialize AutoML
aml = H2OAutoML(max_runtime_secs=600)
aml.train(x=x, y=y, training_frame=data)
lb = aml.leaderboard
print(lb)

# Get the best model
best_model = aml.leader

data = pd.read_csv(csv_file_path)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert the new data to an H2O frame
h2o_new_data = h2o.H2OFrame(X_test)
predictions = best_model.predict(h2o_new_data)

# Print the predictions
print(predictions)

# Shutdown the H2O cluster
h2o.shutdown()