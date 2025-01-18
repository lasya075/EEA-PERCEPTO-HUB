import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the pickled data
with open('data.pkl', 'rb') as f:
    dataset = pickle.load(f)

data = np.asarray(dataset['data'])
labels = np.asarray(dataset['labels'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

# Print the shape of the data to check the number of features
print(f"Shape of training data (X_train): {X_train.shape}")
print(f"Shape of testing data (X_test): {X_test.shape}")

# Initialize models
logistic_regression = LogisticRegression(max_iter=1000)
knn_classifier = KNeighborsClassifier()
random_forest = RandomForestClassifier(class_weight='balanced')

# Train and evaluate Logistic Regression
logistic_regression.fit(X_train, y_train)
lr_predictions = logistic_regression.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)

random_forest.fit(X_train, y_train)
rf_predictions = random_forest.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Evaluate models
models = [random_forest,knn_classifier,logistic_regression]
scores = [{rf_accuracy},{knn_accuracy},{lr_accuracy}]

best_model = models[np.argmax(scores)]
model = best_model.fit(data, labels)

with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f'Best model: {best_model} with accuracy score: {np.max(scores)}')