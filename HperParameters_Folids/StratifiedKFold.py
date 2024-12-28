from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import numpy as np

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Define StratifiedKFold for cross-validation on the entire dataset
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# List to store the accuracy scores for each fold
accuracy_scores = []

# Perform StratifiedKFold cross-validation on the entire dataset
for train_index, val_index in kf.split(X, y):
    # Split the data into train and validation sets for each fold
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]
    
    # Train the model on the training fold
    rf.fit(X_train_fold, y_train_fold)
    
    # Predict on the validation fold
    y_pred = rf.predict(X_val_fold)
    
    # Calculate accuracy for the current fold
    accuracy = accuracy_score(y_val_fold, y_pred)
    accuracy_scores.append(accuracy)

# Calculate the average accuracy across all folds
average_accuracy = np.mean(accuracy_scores)

# Print results
print(f"Cross-validation accuracy scores: {accuracy_scores}")
print(f"Average cross-validation accuracy: {average_accuracy}")
