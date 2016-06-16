import gensim
import logging 
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
			processed.append(filename)
			sentences.append(processed)
	#model = gensim.models.Word2Vec(sentences)
	#model.save('./mymodel')
	print output
	model = gensim.models.Word2Vec.load('./mymodel')
	return input, output


x, y = process_input()
