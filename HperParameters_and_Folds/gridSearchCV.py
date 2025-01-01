from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
X=load_breast_cancer().data
Y=load_breast_cancer().target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


rf=RandomForestClassifier()
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
param_grid={
    'n_estimators':[50,100,200],
    'max_depth':[None,10,20,30],
    'min_samples_split':[2,3,5],
    'min_samples_leaf':[1,2,3,2,4]
}
gridSearch=GridSearchCV(estimator=rf,param_grid=param_grid,cv=cv,n_jobs=-1,verbose=2)

gridSearch.fit(X_train,Y_train)


pred=gridSearch.best_estimator_.predict(X_test)
acc=accuracy_score(Y_test,pred)
print(f" Accuracy:{acc*100}")
print(f" Best params:{gridSearch.best_params_}")
print(f" Best best:{gridSearch.best_estimator_}")


