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
    """Helps Obtain the cost function for any linear regressor"""
    example_count     = x.shape[0]
    squared_error     = (hypothesis_function(theta,x)-y)**2
    sum_squared_error = sum(squared_error)
    cost_value        = sum_squared_error/(2*example_count)
    return cost_value

def hypothesis_function(theta,x):
    """Helps Obtain the hypothesis for any linear regressor"""
    return x.dot(theta)

def feature_normalization(x):
    """Helps with noramlizing feature data"""
    pass

def normal_equation(x,y):
    pass
def gradient_descent(x,y,theta,num_of_iter,alpha):
    """Helps Obtain the optimal value of weighted values by minimizing cost function for any linear regressor"""
    example_count   = x.shape[0]
    cost_values     = np.zeros([num_of_iter,1])
    for k in range(num_of_iter):
        for j in range(theta.size):
            error_factor      = (x.dot(theta)-y)*np.array([x[:,j]]).transpose()
            error_sum_factor  = sum(error_factor)*(alpha/example_count)
            theta[j]          = theta[j] - error_sum_factor
        cost_values[k] = cost_function(x,y,theta)
    return (theta,cost_values) 
            
            
        
        
        
    
    