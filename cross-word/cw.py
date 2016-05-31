from nolearn.lasagne import NeuralNet
from lasagne import layers
from lasagne import nonlinearities
from lasagne.updates import nesterov_momentum

SEARCH_CLUES_PATH = "./cw/"

def process_filename(line):
	if line.strip() != "":
		line = line.rstrip('\n')
		file_id, filename = line.split('\t')
		return file_id, filename

def process_txt(line):
	if line.strip() != "":
		line = line.rstrip('\n')
		clues, url = line.split('\t')
		clues = clues.split(' ')
		return clues

def process_input():
	for line in open(SEARCH_CLUES_PATH + 'list', 'r').readlines():
		file_id, filename = process_filename(line)
		for line2 in open(SEARCH_CLUES_PATH + filename, 'r').readlines():
			print process_txt(line2)
def train():
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

def_predict

process_input()