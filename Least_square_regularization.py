# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 04:06:43 2019

@author: nakul
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from numpy import linalg as LA
from numpy.linalg import inv



# Loading the data
f = open('data3_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]
X = X.reshape(200,1)
Y = Y.reshape(200,1)


lamda = 100

# Appending column of 1's to the X array of data
b = np.ones([200,1],dtype = int)
x = np.hstack((X,b))
print(x.shape)


# Calculating the model parameters
beta = np.matmul(LA.pinv(np.matmul(x.T ,x) + lamda*np.eye(2, dtype=int)) , np.matmul(x.T , Y))
x1 = np.linspace(-150,150)
y1 = (beta[0]*x1+beta[1])


# Plot the line
plt.title('Least Square Estimation with Regularization (data-3)')
plt.plot(x1,y1,'r')
plt.scatter(X,Y,label = 'Least Square Estimation with Regularization (data-1)')
plt.show()
