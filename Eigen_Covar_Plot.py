# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:42:22 2019

@author: nakul
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
n = X.size


# Defining the function for variance
def var(a1):
  variance = np.mean(abs(a1 - np.mean(a1))**2)
  return variance

# Defining the function for covariance
def cov(a2,b2):
  covariance = (n/(n-1)) *  (np.mean(X*Y)-(np.mean(X)*np.mean(Y)) )
  return covariance


Xbar = np.mean(X) 
Ybar = np.mean(Y)

# Writing the elements of the covariance matrx
Sxx = float(var(X))
Sxy = float(cov(X,Y))
Syy = float(var(Y))
Syx = float(cov(Y,X))

# Define covariance matrix
Sigma = np.array([[Sxx ,Sxy] , [Syx, Syy]],dtype = np.float64)

# Finding eigenvalues and eigenvectors
W,V = LA.eig(Sigma)
eigen_vector_1 = V[:,0]
eigen_vector_2 = V[:,1]
origin = [Xbar,Ybar]
V0 = W[0]*V[:,0]
V1 = W[1]*V[:,1]


# Plot the figure
plt.title('Geometric interpretation of the eigenvalues/eigenvectors and the covariance matrix') 
plt.scatter(X,Y,c ='limegreen')
plt.quiver(*origin, V0[0], V0[1], scale=9999)
plt.quiver(*origin, V1[0], V1[1], scale=8000)
#plt.quiver(*origin,V[:,0],V[:,0],d[1],'k','LineWidth',5)
plt.show()
