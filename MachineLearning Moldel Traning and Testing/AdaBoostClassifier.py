import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier  # Import AdaBoost Classifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed  # For parallel processing

# Function to load training data (using parallel processing)
def load_training_data(train_directory):
    def process_image(class_name, img_name):
        img_path = os.path.join(train_directory, class_name, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (32, 32))  # Resize to 32x32 (smaller size for faster processing)
            return img.flatten(), class_name
        return None

    X_train, y_train = [], []
    for class_name in os.listdir(train_directory):
        class_path = os.path.join(train_directory, class_name)
        if os.path.isdir(class_path):
            # Parallelize image loading and processing
            results = Parallel(n_jobs=-1)(delayed(process_image)(class_name, img_name)
                                          for img_name in os.listdir(class_path))
            for result in results:
                if result:
                    X_train.append(result[0])
                    y_train.append(result[1])
    return np.array(X_train), np.array(y_train)

# Function to load test data from CSV and directory
def load_test_data_from_csv(csv_path, test_directory):
    df = pd.read_csv(csv_path, header=None, names=["Filename"])

    def process_image(img_name):
        img_path = os.path.join(test_directory, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (32, 32))  # Resize to 32x32
            return img.flatten()
        return None

    X_test, y_test, filenames = [], [], []
    for _, row in df.iterrows():
        img_name = row["Filename"]
        processed_img = process_image(img_name)
        
        if processed_img is not None:  # Only append if the image was successfully processed
            X_test.append(processed_img)

            if "Expected_Label" in df.columns:
                y_test.append(row["Expected_Label"])
            filenames.append(img_name)

    # Ensure that X_test and y_test have consistent shapes
    X_test = np.array(X_test)
    if y_test:
        y_test = np.array(y_test)
    return X_test, y_test if y_test else None, filenames

# Paths for train, test, and CSV file
train_dir = "./train"
test_dir = "./test"
test_csv_path = "./Testing_set_flower.csv"

# Load training data
X_train, y_train = load_training_data(train_dir)

# Split training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train the AdaBoost model with hyperparameter tuning
# Reduced hyperparameter space for faster tuning
param_grid = {
    'n_estimators': [50, 100],  # Fewer options for n_estimators
    'learning_rate': [0.1, 0.5],  # Fewer options for learning_rate
}

# Initialize AdaBoost classifier
ada_classifier = AdaBoostClassifier(random_state=42)

# GridSearchCV for hyperparameter tuning with 3-fold cross-validation (faster)
grid_search = GridSearchCV(ada_classifier, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_split, y_train_split)

# Get the best model from GridSearchCV
best_ada_classifier = grid_search.best_estimator_

# Evaluate on the validation set
y_val_pred = best_ada_classifier.predict(X_val_split)
val_accuracy = accuracy_score(y_val_split, y_val_pred)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Load test data
X_test, y_test, test_filenames = load_test_data_from_csv(test_csv_path, test_dir)

# Make predictions on the test set
y_test_pred = best_ada_classifier.predict(X_test)

# Display predictions
for img_name, pred_label in zip(test_filenames, y_test_pred):
    print(f"{img_name}: Predicted as {pred_label}")

# If labels exist in the CSV, calculate test accuracy
if y_test is not None:
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Validation Accuracy: 47.44%