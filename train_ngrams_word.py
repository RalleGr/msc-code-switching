import pandas as pd
from models.ngrams.word_ngrams import NGramModel
from tools.utils import write_dict
from tools.utils import is_other
import os
from bs4 import BeautifulSoup
import re
from tools.utils import printStatus
from spacy.lang.en import English
from spacy.lang.es import Spanish
import pickle

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"

n = 2
if n!=2 and n!=3 and n!=4 and n!=5 and n!=6:
	print("n should be 2, 3, 4, 5 or 6")
	exit(1)

# Split dataset by sentences, each sentenced by tokens
def get_tokenized_sentences(lang):
	
	tokenizedFile = []
	# Initialize tokenizer
	nlp = English() if lang is 'en' else Spanish()
	tokenizer = nlp.Defaults.create_tokenizer(nlp)

	# Load data
	printStatus("Creating tokenized sentences from dataset...")
	for root, dirs, files in os.walk('datasets/monolingual-' + lang):
		if ('.DS_Store' in files):
			files.remove('.DS_Store')
		for f in files:
			print(f)
			filepath = os.path.join(root, f)
			file = open(filepath, 'rt', encoding='utf8')
			text = file.read()
			file.close()

			# Clean XML tags
			cleantext = BeautifulSoup(text, "lxml").text

			# Split in sentences
			sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", cleantext)
			
			# Split in tokens
			for s in sentences:
				word_tokens = []
				tokens = list(tokenizer(s))
				for t in tokens:
					t = t.text.lower()
					if (not is_other(t)):
						word_tokens.append(t)

				tokenizedFile.append(word_tokens)

	return tokenizedFile

# tokenized_sentences_en = get_tokenized_sentences('en')
# with open('./dictionaries/word-level/tokenized_sentences_en.p', 'wb') as fp:
# 	pickle.dump(tokenized_sentences_en, fp)

# tokenized_sentences_es = get_tokenized_sentences('es')
# with open('./dictionaries/word-level/tokenized_sentences_es.p', 'wb') as fp:
# 	pickle.dump(tokenized_sentences_es, fp)

tokenized_sentences_en = pd.read_pickle(r'./dictionaries/word-level/tokenized_sentences_en.p')
tokenized_sentences_es = pd.read_pickle(r'./dictionaries/word-level/tokenized_sentences_es.p')

model_en = NGramModel(n)
model_en.train(tokenized_sentences_en)
write_dict(WORD_LEVEL_DICTIONARIES_PATH, model_en.freq_dist, str(n) + '_grams_word_dict_en')

model_es = NGramModel(n)
model_es.train(tokenized_sentences_es)
write_dict(WORD_LEVEL_DICTIONARIES_PATH, model_es.freq_dist, str(n) + '_grams_word_dict_es')