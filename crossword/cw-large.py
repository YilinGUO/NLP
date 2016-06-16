from nolearn.lasagne import NeuralNet
from lasagne import layers
from lasagne import nonlinearities
from lasagne.updates import nesterov_momentum
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import numpy as np

SEARCH_CLUES_PATH = "./cw-large/"
#SEARCH_CLUES_PATH = "/data0/corpora/crossword_corpus/"

def process_filename(line):
	if line.strip() != "":
		line = line.rstrip('\n')
		#file_id, filename = line.split('\t')
		#file_id = int(file_id)
		#return file_id, filename
        return line

def process_txt(line):
	if line.strip() != "":
		line = line.rstrip('\n')
		clues, url = line.split('\t')
		# clues = clues.split(' ')
		return clues

def process_input():
	sentences = []
	output = []
	count = 0
	for line in open(SEARCH_CLUES_PATH + 'random1000.list', 'r').readlines():
		filename = process_filename(line)
		for line2 in open(SEARCH_CLUES_PATH + filename, 'r').readlines():
			sentences.append(process_txt(line2))
			#output.append(filename)
			output.append(count)
		count = count + 1
	vectorizer = CountVectorizer(min_df=1, encoding='latin_1')
	input = vectorizer.fit_transform(sentences).toarray()
	output = np.array(output)
	print output
	return input, output

def split_data(input, output):
	x_train, x_test, y_train, y_test = train_test_split(input, output, test_size = 0.2)
	return x_train, x_test, y_train, y_test

x, y = process_input()
x_train, x_test, y_train, y_test = split_data(x, y)