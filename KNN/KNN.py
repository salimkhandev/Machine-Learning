from sklearn.neighbors import KNeighborsClassifier

# Student data: Age and preference (1 for sports, 0 for music)
X = [[15], [16], [14], [18]]  # Ages
y = [1, 0, 1, 0]  # Preferences (1 = sports, 0 = music)

# New student's age
new_student = [[13]]

# Create a KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Predict the preference for the new student
prediction = knn.predict(new_student)

if prediction == 1:
    print("The new student will probably like sports.")
else:
    print("The new student will probably like music.")



