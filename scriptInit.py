import numpy as np
from sklearn import preprocessing

file="C/Users/ADMIN/Documents/Github/Python-Machine_Learning-Cookbook/Chapter01/data_singlevar.txt"

data                =np.array([[3,-1.5,2,-5.4],[0,4,-0.3,2.1],[1,3.3,-1.9,-4.3]])
data_standardized   =preprocessing.scale(data)
data_scaler         =preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled         =data_scaler.fit_transform(data)
data_normalised     =preprocessing.normalize(data,'l2')
data_binarized      =preprocessing.Binarizer().transform(data)
print(f"\nBinarised Data \n{data_binarized}")
print(f"\nNormalized data \n{data_normalised}")
print(f"\nMin Maxed Values \n{data_scaled}")
print(f"\nMean          \n{data_standardized.mean(axis=0)}")
print(f"\nStd deviation \n{data_standardized.std(axis=0)}")

