import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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
    print("Loaded Test CSV:\n", df.head()) 

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

print("Loading training data...")
X_train, y_train = load_training_data(train_dir)
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
print(f"Training set shape: {X_train_split.shape}, Validation set shape: {X_val_split.shape}")

label_encoder = LabelEncoder()
y_train_split_encoded = label_encoder.fit_transform(y_train_split)
y_val_split_encoded = label_encoder.transform(y_val_split)

param_grid = {
    'max_depth': [3],       
    'learning_rate': [0.1],  
}

print("Running Grid Search...")
grid_search = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                           param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train_split, y_train_split_encoded)

best_xgb_classifier = grid_search.best_estimator_

print(f"Best hyperparameters: {grid_search.best_params_}")

print("Evaluating on the validation set...")
y_val_pred_encoded = best_xgb_classifier.predict(X_val_split)
y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)
val_accuracy = accuracy_score(y_val_split, y_val_pred)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

print("Loading test data...")
X_test, y_test, test_filenames = load_test_data_from_csv(test_csv_path, test_dir)
print(f"Test data shape: {X_test.shape}")

print("Making predictions on the test set...")
y_test_pred_encoded = best_xgb_classifier.predict(X_test)
y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)

print("Test Predictions:")
for img_name, pred_label in zip(test_filenames, y_test_pred):
    print(f"{img_name}: Predicted as {pred_label}")

if y_test is not None:
    y_test_encoded = label_encoder.transform(y_test)
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred_encoded)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    
    
# Validation Accuracy: 55.90%
# Loading test data...