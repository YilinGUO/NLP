
# coding: utf-8

# Import the iris data

# In[250]:

import urllib2
import numpy as np
targetURL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
data = urllib2.urlopen(targetURL)
x1 = []
x2 = []
x3 = []
for line in data:
    if line.strip():
        line = line.split(',')
        if line[4] == 'Iris-setosa\n':
            line = line[0:4]
            line = [float(i) for i in line]
            line.append(-1)
            x1.append(line)
        elif line[4] == 'Iris-versicolor\n':
            line = line[0:4]
            line = [float(i) for i in line]
            line.append(1)
            x2.append(line)
        else:
            line = line[0:4]
            line = [float(i) for i in line]
            line.append(0)
            x3.append(line)


# split the data into training and testing sets

# In[251]:

import pandas as pd
from sklearn.cross_validation import train_test_split
X = np.concatenate((x1, x2))
x_train, x_test = train_test_split(X, test_size = 0.2)


# define the class perceptron

# In[252]:

import random as rd
class Perceptron(object):
    """docstring for Perceptron"""
    def __init__(self):
        super(Perceptron, self).__init__()
        self.w = [rd.random() * 2 - 1 for _ in xrange(2)] # weights
        self.learningRate = 0.1
        self.bias = 0

    def response(self, x):
        """perceptron output"""
        y = x[0] * self.w[0] + x[1] * self.w[1] + self.bias
        if y >= 0:
            return 1
        else:
            return -1
    
    def updateWeights(self, x):
        """
        upates the weights and the bias
        """
        self.w[0] += self.learningRate * x[0] * x[2]
        self.w[1] += self.learningRate * x[1] * x[2]
        self.bias += self.learningRate * x[2]

    def train(self, data):
        """
        trains all the vector in data.
        Every vector in data must three elements.
        the third element(x[2]) must be the label(desired output)
        """
        learned = False
        iteration = 0
        while not learned:
            globalError = 0.0
            for x in data: # for each sample
                r = self.response(x)
                if x[2] != r: # if have a wrong response
                    self.updateWeights(x)
            iteration += 1
            if iteration >= 1000: # stop criteria
                learned = True # stop learing
    
    def decision_boundary(self, x):
        return (-self.bias - x * self.w[0]) / self.w[1]


# use perceptron to predict

# In[253]:

import matplotlib.pyplot as plt
p = Perceptron()
p.train(x_train[:, (0,1,4)])

fig, ax = plt.subplots()

x_train = np.array(x_train)
x_test = np.array(x_test)

R = x_train[x_train[:, 4] == -1]
B = x_train[x_train[:, 4] == 1]

ax.scatter(R[:,0],R[:,1],marker='o', facecolor = 'red', label = "Training Outcome 1")
ax.scatter(B[:,0],B[:,1],marker='o', facecolor = 'blue', label = "Training Outcome -1")
ax.plot(x_train[:, 0], p.decision_boundary(x_train[:, 0]), label='Predicted Decision Boundary')

R2 = x_test[x_test[:, 4] == -1]
B2 = x_test[x_test[:, 4] == 1]

ax.scatter(R2[:,0],R2[:,1], marker='D', facecolor = 'red', label = "Testing Outcome 1")
ax.scatter(B2[:,0],B2[:,1], marker='D', facecolor = 'blue', label = "Testing Outcome -1")

ax.legend(loc="upper left", prop={'size':8})
response = []
for x in x_test[:, (0,1,4)]:
    response.append(p.response(x))
result = list((response == x_test[:,4]))
accuracy = result.count(True) / float(len(result))
print "Accuracy for this model: ",'{0:.0%}'.format(accuracy)
plt.show()
