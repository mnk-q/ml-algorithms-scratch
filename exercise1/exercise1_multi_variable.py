import numpy as np
import matplotlib.pyplot as plt
import os


def featureScaling(x):
    x_norm = x.copy()
    means = np.mean(x_norm,axis=0)
    stdevs = np.std(x_norm, axis=0)
    x_norm = x_norm - means
    x_norm = x_norm/stdevs
    return x_norm, means, stdevs
    
def costFunction(x, y, theta):
    h_theta = x.dot(theta)
    m = y.shape[0]
    J = np.sum((h_theta - y)**2) / (2*m)
    return J
data = np.loadtxt(os.path.join('exercise1','Data', 'ex1data2.txt'), delimiter=',')

x = data[:,:2]
y = data[:,2]
m = y.shape
x_norm ,mu, sigma = featureScaling(x)
#We will do some preprocessing i.e. Feature Normalization
# The scaling should be done, so that mean is 0 and standard deviation is 1, for each column of x.
# This can be done by Subtracting mean from every value, and dividing every value with std, for each feature

# The values, which are going to be predicted, must be scaled using these means and standard deviations that were used to scale the values.
# After scaling we should add the 1, which is going to multiplied with theta0

#Gradient Descent in Action
X = np.insert(x_norm, 0,1, axis=1)
theta = np.zeros(3)
J_history=[]
alpha = 0.1
for i in range(1500):
    h_theta = X.dot(theta)
    j1 = theta - alpha *(X.transpose().dot((h_theta - y)))/m
    theta = j1
    J_history.append(costFunction(X, y, theta))

test = np.array([1500,3])
test  = (test - mu)/sigma
test = np.insert(test,0,1)
predict = np.dot(test, theta)
print('For 1500 sq feet with 3 bedrooms, sell price should be {:.2f}\n'.format(predict))