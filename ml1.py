from scipy.io import loadmat
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix,classification_report
import itertools 
import numpy as np
import matplotlib.pyplot as plt

def displayConfusionMatrix(cm,cmap=plt.cm.Spectral):
    classes_x=["Other Number","Number 5"]
    classes_y=["Other Number Group","Number 5 Group"]
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    trick_marks=np.arange(len(classes_x))
    plt.xticks(trick_marks,classes_x)
    plt.yticks(trick_marks,classes_y)
    thresh=cm.max()/2
    for i , j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],'d'),
        horizontalalignment='center',
        color='white' if cm[i,j]>thresh else 'black')

    plt.tight_layout()
    plt.ylabel('Actually')
    plt.xlabel('Prediction')
    plt.show()

def displayImage(x):
    plt.imshow(
        x.reshape(28,28),
        cmap=plt.cm.binary,
        interpolation="nearest")
    plt.show()

def displayPredict(clf, actually_y,x_test):
    print("Actually:",actually_y)
    print("Prediction:",clf.predict([x_test])[0])   
 
mnist_raw = loadmat("mnist-original.mat")
mnist = {
   "data":mnist_raw["data"].T,
   "target":mnist_raw["label"][0]
}

x,y = mnist["data"],mnist["target"]

xtrain,xtest,ytrain,ytest = x[:60000], x[60000:], y[:60000],y[60000:] 

#class 0, class not 0
#ข้อมูล 1 ค่า -> data model ->class 0 ? true: false 
predict_number = 6000
#create boolean array if the value equals zero
ytrain_5 = (ytrain==5)
ytest_5 = (ytest==5)

sgd_clf = SGDClassifier() 
"""สร้าง Object SGD"""
sgd_clf.fit(xtrain,ytrain_5)
"""เทรนข้อมูลทั้งหมดโดยเช็ค class""" 

ytrain_pred = cross_val_predict(sgd_clf,xtrain,ytrain_5,cv=3)
cm=confusion_matrix(ytrain_5,ytrain_pred)

ytest_pred = sgd_clf.predict(xtest)

classes = ["Other numbers","Number 5"]
print(classification_report(ytest_pred,ytest_5,target_names=classes))