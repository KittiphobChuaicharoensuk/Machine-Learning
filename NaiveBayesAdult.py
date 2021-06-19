from sklearn.preprocessing import LabelEncoder
import pandas as pd

adult_income_datasets = pd.read_csv("adult.csv")


def cleandata(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            LE = LabelEncoder()