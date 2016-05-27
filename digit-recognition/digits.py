
# coding: utf-8

# load the digit dataset from sklearn, split the dataset into 80% training and 20% testing

# In[2]:

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits['images'].reshape(1797,64), digits['target'], test_size = 0.2)


# #### Random Forest Classifier

# In[3]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf_rf = RandomForestClassifier()
clf_rf.fit(x_train, y_train)
y_pred_rf = clf_rf.predict(x_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print "random forest accuracy: ", acc_rf


# #### Stochastic Gradient Descent

# In[4]:

from sklearn.linear_model import SGDClassifier
clf_sgd = SGDClassifier()
clf_sgd.fit(x_train, y_train)
y_pred_sgd = clf_sgd.predict(x_test)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
print "stochastic gradient descent accuracy: ",acc_sgd


# #### Support Vector Machine

# In[5]:

from sklearn.svm import LinearSVC
clf_svm = LinearSVC()
clf_svm.fit(x_train, y_train)
y_pred_svm = clf_svm.predict(x_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print "Linear SVM accuracy: ",acc_svm


# #### Nearest Neighbors

# In[6]:

from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier()
clf_knn.fit(x_train, y_train)
y_pred_knn = clf_knn.predict(x_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print "nearest neighbors accuracy: ",acc_knn


# #### Neural Network

# In[7]:

from nolearn.lasagne import NeuralNet
from lasagne import layers
from lasagne import nonlinearities
from lasagne.updates import nesterov_momentum
clf_nn = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 64),  # 784 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=10,  # 10 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    
    max_epochs=10,  # we want to train this many epochs
    verbose=1,
    )
clf_nn.fit(x_train, y_train)

