import numpy as np
from scipy import linalg
import random as rd
import matplotlib.pyplot as plt
def test_set(a,b, no_samples, intercept, slope, variance, offset):
    #First we grab some random x-values in the range [a,b].
    X1 = [rd.uniform(a,b) for i in range(no_samples)]
    #Sorting them seems to make the plots work better later on.
    X1.sort()
    #Now we define our linear function that will underly our synthetic dataset: the true decision   boundary.
    def g(x):
        return intercept + slope*x
    #We create two classes of points R,B. R points are given by g(x) plus a Gaussian noise term with positive mean,
    #B points are given by g(x) minus a Gaussian norm term with positive mean.
    R_turn=True
    X2=[] #This will hold the y-coordinates of our data.
    Y=[]  #This will hold the classification of our data.
    for x in X1:
        x2=g(x)
        if R_turn:
            x2 += rd.gauss(offset, variance)
            Y.append(1)
        else:
            x2 -= rd.gauss(offset, variance)
            Y.append(-1)
        R_turn=not R_turn
        X2.append(x2)
    #Now we combine the input data into a single matrix.
    X1 = np.array([X1]).T
    X2 = np.array([X2]).T
    X = np.hstack((X1, X2))
    #Now we return the input, output, and true decision boundary.
    return [ X,np.array(Y).T, map(g, X1)]


X,Y,g = test_set(a=0, b=5, no_samples=200, intercept=10, slope=0, variance=0.5, offset=1)
fig, ax = plt.subplots()
R =  X[::2] #Even terms
B =  X[1::2] #Odd terms
 
ax.scatter(R[:,0],R[:,1],marker='o', facecolor='red', label="Outcome 1")
ax.scatter(B[:,0],B[:,1],marker='o', facecolor='blue', label="Outcome 0")
ax.plot(X[:,0], g, color='green', label="True Decision Boundary")
ax.legend(loc="upper left", prop={'size':8})
#plt.show()

def perceptron(X,y, learning_rate=0.1, max_iter=1000):
    #Expecting X = n x m inputs, y = n x 1 outputs taking values in -1,1
    m=X.shape[1]
    n=X.shape[0]
    weight = np.zeros((m))
    bias = 0
    n_iter=0
    index = list(range(n))
    while n_iter <= max_iter:
        rd.shuffle(index)
        for row in index:
            if (X[row,:].dot(weight)+bias)*y[row] <=0: #Misclassified point.
                weight += learning_rate*X[row,:]*y[row]
                bias += learning_rate*y[row]
        n_iter+=1
    return {'weight':weight, 'bias':bias}

result= perceptron(X,Y,0.1)
def decision_boundary(x, weight, bias):
    return (-bias - x*weight[0])/weight[1]
 
fig, ax = plt.subplots()
R =  X[::2] #Even terms
B =  X[1::2] #Odd terms
 
ax.scatter(R[:,0],R[:,1],marker='o', facecolor='red', label="Outcome 1")
ax.scatter(B[:,0],B[:,1],marker='o', facecolor='blue', label="Outcome -1")
ax.plot(X[:,0], g, color='green', label="True Decision Boundary")
ax.plot(X[:,0], [decision_boundary(x, result['weight'], result['bias']) for x in X[:,0]],
        label='Predicted Decision Boundary')
ax.legend(loc="upper left", prop={'size':8})
plt.show()