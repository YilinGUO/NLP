import gensim
import logging
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SEARCH_CLUES_PATH = "./cw/"

def process_filename(line):
	if line.strip() != "":
		line = line.rstrip('\n')
		file_id, filename = line.split('\t')
		filename = filename[:-4]
		file_id = int(file_id)
		return file_id, filename

def process_txt(line):
	if line.strip() != "":
		line = line.rstrip('\n')
		clues, url = line.split('\t')
		clues = clues.split()
		return clues

def process_input():
	sentences = []
	output = []
	for line in open(SEARCH_CLUES_PATH + 'list', 'r').readlines():
		file_id, filename = process_filename(line)
		output.append(filename)
		for line2 in open(SEARCH_CLUES_PATH + filename + '.txt', 'r').readlines():
			processed = process_txt(line2)
			processed.append(filename) ###
			sentences.append(processed)
	train, test = train_test_split(sentences, test_size = 0.2)
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,  preprocessor = None, stop_words = None)
	train_data_features = vectorizer.fit_transform(sentences)
	vocab = vectorizer.get_feature_names()
	print vocab
	return train, test, output


train, test, output = process_input()
model = gensim.models.Word2Vec(train)
model.save('./mymodel')
vocab = model.scan_vocab(train)
print vocab
#model = gensim.models.Word2Vec.load('./mymodel')
#print test[0]
#print test[0][1],output[1]
#print model.similarity(test[0][1],output[1])
