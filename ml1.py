from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

df = pd.read_csv("diabetes.csv")

x=df.drop("Outcome",axis=1) 
"""1 = column, 0 = row"""
y=df["Outcome"].values
"""เก็บค่าวัดผลเบาหวานทั้งหมด"""

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.4,random_state=0)

#find k to build the matching model
k=np.arange(1,9)