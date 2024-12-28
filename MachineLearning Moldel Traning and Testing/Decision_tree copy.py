from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Initialize KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(knn, X, y, cv=5)

# Print the results
print("Cross-validation scores for each fold:", cv_scores)
print("Average cross-validation score:", cv_scores.mean())
