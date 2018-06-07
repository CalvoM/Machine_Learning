import numpy as np
import pickle
from sklearn import preprocessing,linear_model
import sklearn.metrics as sm
import sys
import matplotlib.pyplot as plt

output_model_file='saved_model.pkl'
file="C:/Users/ADMIN/Documents/Github/Python-Machine-Learning-Cookbook/Chapter01/data_singlevar.txt"
x=[]
y=[]

with open(file,'r') as sourceFile:
    for line in sourceFile:
        xt,yt=[float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)

num_training    =int(0.8*len(x))
num_testing     =len(x)-num_training

x_training      =np.array(x[:num_training]).reshape((num_training,1))
y_training      =np.array(y[:num_training])

x_test          =np.array(x[num_training:]).reshape((num_testing,1))
y_test          =np.array(y[num_training:])

linear_regressor=linear_model.LinearRegression()
linear_regressor.fit(x_training,y_training)

y_train_pred    =linear_regressor.predict(x_training)

y_test_pred     =linear_regressor.predict(x_test)

mean_absolute   =round(sm.mean_absolute_error(y_test,y_test_pred),2)
mean_squared    =round(sm.mean_squared_error(y_test,y_test_pred))

print(f"Mean absolute {mean_absolute}")
print(f"Mean squared error {mean_squared}")

with open(output_model_file,'wb') as file:
    pickle.dump(linear_regressor,file)
# plt.figure()
# plt.scatter(x_test,y_test,color='red')
# plt.plot(x_test,y_test_pred,color='green')
# plt.show()
