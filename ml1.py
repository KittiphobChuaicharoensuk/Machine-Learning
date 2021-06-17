from scipy.io import loadmat
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

def displayImage(x):
    plt.imshow(
        x.reshape(28,28),
        cmap=plt.cm.binary,
        interpolation="nearest")
    plt.show()

def displayPredict(clf, actually_y,x_test):
    print("Actually:",actually_y)
    print("Prediction:",clf.predict([x_test][0]))   
 
mnist_raw = loadmat("mnist-original.mat")
mnist = {
   "data":mnist_raw["data"].T,
   "target":mnist_raw["label"][0]
}

x,y = mnist["data"],mnist["target"]

xtrain,xtest,ytrain,ytest = x[:60000], x[60000:], y[:60000],y[60000:] 

#class 0, class not 0
#ข้อมูล 1 ค่า -> data model ->class 0 ? true: false 
predict_number = 5000
#create boolean array if the value equals zero
ytrain_5 = (ytrain==5)
ytest_5 = (ytest==5)


sgd_clf = SGDClassifier() 
"""สร้าง Object SGD"""
sgd_clf.fit(xtrain,ytrain_5)
"""เทรนข้อมูลทั้งหมดโดยเช็ค class""" 

score = cross_val_score(sgd_clf,xtrain,ytrain_5,cv=3,scoring="accuracy")
print(score)