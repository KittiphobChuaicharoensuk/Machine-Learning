from pandas.core.frame import DataFrame
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
# make_blobs imitate or model the sample
# how many rows or column

x,y=make_blobs(n_samples=100,n_features=10)

# x stores sample+attribute 
# y stores feature

print(f"Before {x.shape}")
pca=PCA(n_components=4)
pca.fit(x)

df = DataFrame({"var":  })

# x=pca.transform(x)
# print(f"After: {x.shape}")