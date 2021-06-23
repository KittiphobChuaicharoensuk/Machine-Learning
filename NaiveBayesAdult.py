from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
import pandas as pd

def cleandata(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            LE = LabelEncoder()
            dataset[column]=LE.fit_transform(dataset[column])
    return dataset

def split_feature_class(dataset,feature):
    ft=dataset.drop(feature,axis=1)
    label=dataset[feature].copy()
    return ft,label


adult_income_datasets = pd.read_csv("adult.csv")
clean_dataset=cleandata(adult_income_datasets)

# train/test split
training_set,test_set = tts(clean_dataset,test_size=0.2,)

# train 
train_feature,train_label=split_feature_class(training_set,"income")

# test
test_feature, test_label = split_feature_class(test_set,"income")

# model
model = GaussianNB()
model.fit(train_feature,train_label)

#predict
test_label_pred = model.predict(test_feature)

#evaluate the result
print(f"\n\nAccuracy: {accuracy_score(test_label,test_label_pred)*100} %\n")
print(f"MAE: {mean_absolute_error(test_label,test_label_pred)}\n")
print(f"F1_Score: {f1_score(test_label,test_label_pred)*100} %\n\n")


