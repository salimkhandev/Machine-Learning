# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Load the iris dataset
iris = datasets.load_iris()

# The iris dataset contains:
# - Features: Sepal Length, Sepal Width, Petal Length, Petal Width
# - Target: Species (Setosa, Versicolor, Virginica)
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target labels (0 = Setosa, 1 = Versicolor, 2 = Virginica)
                                                        
# Split the dataset into training (80%) and testing (20%) sets
features_train, features_test, labels_train, labels_test = train_test_split(X, y)

# Create a KNN classifier with k=3 (K-Nearest Neighbors model)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training data (features and target labels)
knn.fit(features_train, labels_train)

# Predict the species (target labels) for the test set
predicted_labels = knn.predict(features_test)

# Generate a confusion matrix to compare predictions to actual species
conf_matrix = confusion_matrix(labels_test, predicted_labels)

# Print the confusion matrix and classification report
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(labels_test, predicted_labels, target_names=iris.target_names))

# Plot the confusion matrix with species names
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Iris Dataset")
plt.show()
