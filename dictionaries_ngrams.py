import pandas as pd
from ngrams.word_ngrams import NGramModel
from tools.utils import write_dict
from tools.utils import is_other
import sys
import os
from bs4 import BeautifulSoup
import re
from tools.utils import printStatus
from spacy.lang.en import English
from spacy.lang.es import Spanish
import pickle

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"
CHAR_LEVEL_DICTIONARIES_PATH = "./dictionaries/character-level/"

# Get words dictionaries
frequency_en_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH+'frequency_dict_en.csv',encoding='utf-16')
frequency_en_dict = frequency_en_df.set_index('word')['frequency'].to_dict()

frequency_es_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH+'frequency_dict_es.csv',encoding='utf-16')
frequency_es_dict = frequency_es_df.set_index('word')['frequency'].to_dict()

# frequency_other_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH+'frequency_dict_other.csv',encoding='utf-16')
# frequency_other_dict = frequency_other_df.set_index('word')['frequency'].to_dict()

# Create ngrams frequency dictionaries
n = 2
if n!=2 and n!=3 and n!=4 and n!=5 and n!=6:
	print("n should be 2, 3, 4, 5 or 6")
	exit(1)

# model_en = NGramModel(n)
# model_en.train(frequency_en_dict)
# write_dict(CHAR_LEVEL_DICTIONARIES_PATH, model_en.freq_dist, str(n) + '_grams_dict_en')

# model_es = NGramModel(n)
# model_es.train(frequency_es_dict)
# write_dict(CHAR_LEVEL_DICTIONARIES_PATH, model_es.freq_dist, str(n) + '_grams_dict_es')

# model_other = NGramModel(n)
# model_other.train(frequency_other_dict)
# write_dict(CHAR_LEVEL_DICTIONARIES_PATH, model_other.freq_dist, str(n) + '_grams_dict_other')

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
# with open('tokenized_sentences_en.p', 'wb') as fp:
# 	pickle.dump(tokenized_sentences_en, fp)

# tokenized_sentences_es = get_tokenized_sentences('es')
# with open('tokenized_sentences_es.p', 'wb') as fp:
# 	pickle.dump(tokenized_sentences_es, fp)

tokenized_sentences_en = pd.read_pickle(r'tokenized_sentences_en.p')
tokenized_sentences_es = pd.read_pickle(r'tokenized_sentences_es.p')

model_en = NGramModel(n)
model_en.train(tokenized_sentences_en)
write_dict(WORD_LEVEL_DICTIONARIES_PATH, model_en.freq_dist, str(n) + '_grams_word_dict_en')

model_es = NGramModel(n)
model_es.train(tokenized_sentences_es)
write_dict(WORD_LEVEL_DICTIONARIES_PATH, model_es.freq_dist, str(n) + '_grams_word_dict_es')