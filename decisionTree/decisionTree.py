# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Load the Iris dataset
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# Make predictions
dtc_predict = dtc.predict(X_test)

# Calculate accuracy and print confusion matrix
acc = accuracy_score(y_test, dtc_predict)
print("The Confusion Matrix\n", confusion_matrix(y_test, dtc_predict))
print(f"The accuracy is {acc}")

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dtc, filled=True, feature_names=iris_data.feature_names, class_names=iris_data.target_names)
plt.title("Decision Tree for Iris Dataset")
plt.show()

# Feature importance
important_features = dtc.feature_importances_
print(important_features * 100)

feature_names = iris_data.feature_names
importances = dtc.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# Additional metrics
print("Maximum depth of the tree is", dtc.get_depth())
print("Number of leaf nodes", dtc.get_n_leaves())
print("Criterion used", dtc.criterion)

decision_path = dtc.decision_path(X_test)
decision_path = decision_path.toarray()
print(decision_path[:5])
