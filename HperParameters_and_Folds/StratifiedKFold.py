from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data = load_breast_cancer()
X = data.data
Y = data.target

# Initialize classifier and Stratified K-Fold
rf = RandomForestClassifier(n_estimators=10, random_state=42)
kf = StratifiedKFold(n_splits=5 ,random_state=42,shuffle=True)

# List to store accuracy scores
accuracy = []

# Stratified K-Fold cross-validation
for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    # Train and evaluate the model
    rf.fit(X_train, Y_train)
    pred = rf.predict(X_test)
    acc = accuracy_score(Y_test, pred)
    accuracy.append(acc)

# Print results
print(f'Accuracy of each fold: {accuracy}')
print(f'Overall Accuracy: {np.mean(accuracy)}')
