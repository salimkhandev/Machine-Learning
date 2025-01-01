from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

X=load_breast_cancer().data
Y=load_breast_cancer().target

rf=RandomForestClassifier(n_estimators=10,random_state=42)

score=cross_val_score(rf,X,Y,cv=5)

print(score)
print(score.mean())