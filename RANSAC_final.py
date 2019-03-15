
import numpy as np
import matplotlib.pyplot as plt
import pickle
from numpy import linalg as LA



# Loading the data (Rename the pickle file name for different data)
f = open('data1_new.pkl','rb')
data1 = pickle.load(f)
f.close()
da = np.asarray(data1)
X = da[:,0]
Y = da[:,1]


# Function for calculating the slope and intercept of line
# The parameter to function is the 2x2 array(2 data points)

def slope_intercept(arr):
    rs_x = arr[:,0]
    rs_y = arr[:,1]
    slope = (rs_y[1]-rs_y[0])/(rs_x[1]-rs_x[0])
    intercept = rs_y[1]-slope*rs_x[1]
    return slope,intercept
    
# Function for calculating the distance of point from the line
def calc_dist_from_line(p,param1,param2):
    dist = 0
    dist = (abs(param2 + (param1*p[0])-p[1]))/np.sqrt(1 + param1**2)
    return dist

T = 10                     #Threshold of distance
max_iter=400               # maximum number of iterations
best_a = np.array([])      
best_inl = 0               # No. of inliers for best estimated model

for i in np.arange(max_iter):
    inl_num = 0
    a =  np.random.choice(da.shape[0],2, replace =True)
    num = da[a,:]
    m,c = slope_intercept(num)
       
    for dist in da:
        d = calc_dist_from_line(dist,m,c)
        if d<=T:
            inl_num += 1
    if best_inl<=inl_num:
        best_inl = inl_num
        best_a = num
      

print("The number of inliers for best model are" + " " +  str(best_inl))


# Plot the figure
plt.title('Outlier rejection with RANSAC')
plt.scatter(X,Y,c ='limegreen')
a,b = slope_intercept(best_a)
x = np.linspace(-150,150)
y = (a*x+b)
y1 = (a*x+(b+10))
y2 = (a*x+(b-10))
plt.plot(x,y,'r')
plt.plot(x,y1,'c')
plt.plot(x,y2,'c')