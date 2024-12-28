import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def load_training_data(train_directory):
    X_train, y_train = [], []
    for class_name in os.listdir(train_directory):
        class_path = os.path.join(train_directory, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    X_train.append(img.flatten())
                    y_train.append(class_name)
    return np.array(X_train), np.array(y_train)

def load_test_data_from_csv(csv_path, test_directory):
    df = pd.read_csv(csv_path, header=None, names=["Filename"])
    X_test, y_test, filenames = [], [], []
    for _, row in df.iterrows():
        img_name = row["Filename"]
        img_path = os.path.join(test_directory, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            X_test.append(img.flatten())
            filenames.append(img_name)
            if "Expected_Label" in df.columns:
                y_test.append(row["Expected_Label"])
    return np.array(X_test), np.array(y_test) if y_test else None, filenames

train_dir = "./train"
test_dir = "./test"
test_csv_path = "./Testing_set_flower.csv"

X_train, y_train = load_training_data(train_dir)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

nb_classifier = GaussianNB()

grid_search = GridSearchCV(nb_classifier, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train_split, y_train_split)

best_nb = grid_search.best_estimator_

y_val_pred = best_nb.predict(X_val_split)
val_accuracy = accuracy_score(y_val_split, y_val_pred)

X_test, y_test, test_filenames = load_test_data_from_csv(test_csv_path, test_dir)

y_test_pred = best_nb.predict(X_test)

for img_name, pred_label in zip(test_filenames, y_test_pred):
    print(f"{img_name}: Predicted as {pred_label}")

if y_test is not None:
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Validation Accuracy: 46.10%
# Loading test data...
# Loaded Test CSV: