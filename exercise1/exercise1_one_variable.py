import numpy as np
import os
from matplotlib import pyplot as plt

def costFunction(x,y,theta):
    h_theta = x.dot(theta)
    m = y.size

    J = np.sum((h_theta - y)**2) / (2*m)
    return J

data = np.loadtxt(os.path.join('exercise1','Data', 'ex1data1.txt'), delimiter=',')

x = data[:,0]
y = data[:,1]


m = y.size
x= np.stack([np.ones(m), x], axis=1)

theta = np.zeros(2)

alpha = 0.01
num_iters = 1500
J_history = []

#Gradient Descent Starts Here
for i in range(num_iters):
        h_theta = x.dot(theta)
        j1 = theta - alpha *(x.transpose().dot((h_theta - y)))/m
        theta = j1
        J_history.append(costFunction(x, y, theta))
        
plt.plot(x[:, 1], y, 'ro', ms=10, mec='k')

plt.plot(x[:, 1], np.dot(x, theta), '-')
plt.legend(['Training data', 'Linear regression'])
plt.show()


predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1*10000))