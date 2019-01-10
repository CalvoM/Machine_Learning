import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(feature_importance,title,feature_names):
    """Helps calculate and plot features that have great influence on final output"""
    feature_importance = 100.0 * (feature_importance/sum(feature_importance))
    index_sorted = np.flipud(np.argsort(feature_importance))
    pos = np.arange(max(index_sorted)+1) + 0.5
    plt.figure()
    plt.bar(pos,feature_importance[index_sorted])
    plt.xticks(pos,feature_names[index_sorted])
    plt.ylabel("RELATIVE IMPORTANCE")
    plt.title(title)
    plt.show()

def cost_function(x,y,theta):
    pass

def hypothesis_function(theta,x):
    pass

def feature_normalization(x):
    pass

def normal_equation(x,y):
    pass