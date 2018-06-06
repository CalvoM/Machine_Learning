import numpy as np
from sklearn import preprocessing
import sys

file="C/Users/ADMIN/Documents/Github/Python-Machine_Learning-Cookbook/Chapter01/data_singlevar.txt"
x=[]
y=[]

with open(file,r) as sourceFile:
    for line in sourceFile:
        xt,yt=[float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)
