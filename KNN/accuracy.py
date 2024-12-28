from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
myData=load_iris().data 
myTarget=load_iris().target


myDataTrain,myDataTest,myTargetTrain,myTargetTest=train_test_split(myData,myTarget)

# print(len(myDataTrain))
# print(len(myDataTest))
# print(myTarget)

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(myDataTrain,myTargetTrain)
knnPredict=knn.predict(myDataTest)
print(knnPredict)
print(myTargetTest)

import pandas as pd

df=pd.DataFrame({'Actual':myTargetTest,'Predicted':knnPredict})


print(df)

acc=accuracy_score(myTargetTest,knnPredict)
print("Accuracy:",acc)

conf=confusion_matrix(myTargetTest,knnPredict)

print("Confusion Matrix:\n",conf)

# now calculating the per class accuracy
# for this we will import numpy library
import numpy as np
# testing diagonal elements of confusion matrix
conDiag=np.diag(conf)
# sumOfDiag=sum(conDiag)

# # summing up the rows of confusion matrix
# print(sumOfDiag)
# calculating per class accuracy
per_class_acc=np.diag(conf)/np.sum(conf,axis=1)
print("Per Class Accuracy:",per_class_acc)

cm = pd.DataFrame(conf,index=load_iris().target_names,columns=load_iris().target_names)


print(cm)
print(conDiag)