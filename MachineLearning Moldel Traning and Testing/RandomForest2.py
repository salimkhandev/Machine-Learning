from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Define the model and parameter grid
rf = RandomForestClassifier()

param_grid = {
    'n_estimators': [3, 5, 10],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Initialize GridSearchCV
gsc = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)  # Set verbose to 1 for more details

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
gsc.fit(X_train, y_train)

# Get the best parameters from GridSearchCV
bestParams = gsc.best_params_
print("Best Parameters: ", bestParams)

# Make predictions
y_predict = gsc.predict(X_test)

# Evaluate the model
acc = accuracy_score(y_test, y_predict)
cf = confusion_matrix(y_test, y_predict)

# Print the results
print("Accuracy: ", acc)
print("Confusion Matrix: \n", cf)
