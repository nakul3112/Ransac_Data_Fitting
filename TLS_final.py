# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:41:50 2019

@author: ishan
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from numpy import linalg as LA


# Loading the data(Rename the pickle file name for different data)
f = open('data1_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]

# Storing the number of data-points
n = X.size

 
# Calculating the model parameters for Total Least Square technique
B = 1/2 * (((sum((Y)**2)-(n*(np.mean(Y))**2)) - (sum((X)**2)-(n*(np.mean(X))**2))) / (n*np.mean(X)*np.mean(Y) - sum(X*Y)))
b = -B + np.sqrt((B)**2 + 1)
a = np.mean(Y) - b*np.mean(X)

# Writing the equation of line
Y_vals = [b*x + a for x in X]


# Plot the figure
plt.title('Total least square ')
plt.scatter(X,Y,c ='r')
plt.plot(X,Y_vals, label = 'Total Least Square line fit')
plt.legend()
plt.show()
