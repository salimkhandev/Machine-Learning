import numpy as np
from sklearn.model_selection import KFold

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 1, 0, 1, 0, 1])

kf = KFold(n_splits=3)  # 3-fold cross-validation
for train_index, val_index in kf.split(X, y):
    print("Train indices:", train_index)
    print("Validation indices:", val_index)
