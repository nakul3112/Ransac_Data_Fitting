# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:46:08 2019

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

# Defining the function for variance
def var(a1):
  variance = np.mean(abs(a1 - np.mean(a1))**2)
  return variance

# Defining the function for covariance
def cov(a2,b2):
  covariance = (n/(n-1)) *  (np.mean(X*Y)-(np.mean(X)*np.mean(Y)) )
  return covariance
 
 
# Calculating the model parameters for Ordinary Least Square technique  
Xbar = np.mean(X) 
Ybar = np.mean(Y)
Sxx = float(var(X))
Sxy = float(cov(X,Y))
beta1 = Sxy / Sxx
beta2 = Ybar - (beta1*Xbar)

# Writing the equation of  line 
y_vals = [beta1 * i + beta2 for i in X]


# Plot the figure
plt.title('Ordinary Least Squares')
plt.scatter(X,Y,c ='r')
plt.plot(X,y_vals,label = 'Ordinary Least Square line fit')
plt.legend()
plt.show()