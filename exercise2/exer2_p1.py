import os
import numpy as np 
from matplotlib import pyplot as plt


def sigmoid(h):
    return (1/(1+np.exp(-h)))



data = np.loadtxt(os.path.join('exercise2','Data', 'ex2data1.txt'), delimiter=',')
X, y = data[:, 0:2], data[:, 2]
# pos = y == 1
# neg = y == 0
m = X.shape[0]
# # Plot Examples
# plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
# plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
# plt.show()
X = np.insert(X,0,1,axis=1)
theta = np.zeros(3)
h = X.dot(theta)

h_theta = sigmoid(h)

J_theta = -1*np.sum((y*np.log(h_theta) + (1-y)*np.log(1-h_theta)))/m
alpha = 0.055
grad = (1.0/m)* X.transpose().dot(h_theta - y)  
print(grad)