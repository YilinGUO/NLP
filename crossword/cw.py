from nolearn.lasagne import NeuralNet
from lasagne import layers
from lasagne import nonlinearities
from lasagne.updates import nesterov_momentum
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import time
from functools import wraps
 
 
def fn_timer(function):
	@wraps(function)
	def function_timer(*args, **kwargs):
		t0 = time.time()
		result = function(*args, **kwargs)
		t1 = time.time()
		print ("Total time running %s: %s seconds" %
			   (function.func_name, str(t1-t0))
			   )
		return result
	return function_timer

SEARCH_CLUES_PATH = "./cw/"

def process_filename(line):
	if line.strip() != "":
		line = line.rstrip('\n')
		file_id, filename = line.split('\t')
		file_id = int(file_id)
		return file_id, filename

def process_txt(line):
	if line.strip() != "":
		line = line.rstrip('\n')
		clues, url = line.split('\t')
		# clues = clues.split(' ')
		return clues

def process_input():
	sentences = []
	output = []
	for line in open(SEARCH_CLUES_PATH + 'list', 'r').readlines():
		file_id, filename = process_filename(line)
		for line2 in open(SEARCH_CLUES_PATH + filename, 'r').readlines():
			sentences.append(process_txt(line2))
			output.append(file_id)
	vectorizer = CountVectorizer(min_df=1, encoding='cp1252')
	input = vectorizer.fit_transform(sentences).toarray()
	output = np.array(output)
	return input, output

def split_data(input, output):
	x_train, x_test, y_train, y_test = train_test_split(input, output, test_size = 0.2)
	return x_train, x_test, y_train, y_test

@fn_timer
def train(x_train, y_train):
	clf_nn = NeuralNet(
		layers=[  # three layers: one hidden layer
			('input', layers.InputLayer),
			('hidden1', layers.DenseLayer),
			('hidden2', layers.DenseLayer),
			('output', layers.DenseLayer),
			],
		# layer parameters:
		input_shape=(None, 2538),  # 784 input pixels per batch
		hidden1_num_units=100,  # number of units in hidden layer
		hidden2_num_units=100,
		output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
		output_num_units=10,  # 10 target values

		# optimization method:
		update=nesterov_momentum,
		update_learning_rate=0.01,
		update_momentum=0.9,
		
		max_epochs=50,  # we want to train this many epochs
		verbose=1,
		)
	clf_nn.fit(x_train, y_train)
	return clf_nn

def test(clf_nn, x_test):
	return clf_nn.predict(x_test)

x, y = process_input()
x_train, x_test, y_train, y_test = split_data(x, y)
print x_train.shape
clf_nn = train(x_train, y_train)
y_pred_nn = test(clf_nn, x_test)
acc_nn = accuracy_score(y_test, y_pred_nn)
print "neural network accuracy: ", acc_nn