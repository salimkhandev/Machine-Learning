import numpy as np
confusionMatrix=[[80,2,0],
           [0,80,10],
           [0,0,90]]
class_base_acc=np.diag(confusionMatrix)/np.sum(confusionMatrix,axis=1)
print(class_base_acc*100)

# overall accuracy of model

overAllAcc=np.diag(confusionMatrix)
sumOfDiagnal=np.sum(overAllAcc)
overAllArrarySum=np.sum(confusionMatrix)
overAllPercentage=(sumOfDiagnal/overAllArrarySum)*100
print("the accuracy of the model is ",overAllPercentage)