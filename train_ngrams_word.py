import pandas as pd
from models.ngrams.word_ngrams import NGramModel
from tools.utils import write_dict
from tools.utils import is_other
from tools.utils import print_status
from langs import langs
import os
from bs4 import BeautifulSoup
import re
import spacy
import pickle
import sys
import importlib

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"

# Split dataset by sentences, each sentence by tokens
def get_tokenized_sentences(lang_code, lang_name):
	
	tokenizedFile = []
	# Initialize tokenizer
	module = importlib.util.find_spec(lang_code, package="spacy.lang")
	nlp = getattr(spacy.lang, lang_name)() if module is not None else spacy.language.Language()
	tokenizer = nlp.Defaults.create_tokenizer(nlp)

	# Load data
	print_status("Creating tokenized sentences from dataset...")
	for root, dirs, files in os.walk('datasets/monolingual-' + lang_code):
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

# Get language code from keyboard
if len(sys.argv) == 1:
	print("Please give two letter language codes as arg, for example en es")
	print("Please give n value")
	exit(1)

# Lang 1
lang1 = sys.argv[1]
lang1_code = langs()[lang1]['code']
lang1_name = langs()[lang1]['name']

# Lang 2
lang2 = sys.argv[2]
lang2_code = langs()[lang2]['code']
lang2_name = langs()[lang2]['name']

# If tokenized sentences exist, read them, otherwise create them
lang1_path = WORD_LEVEL_DICTIONARIES_PATH + 'tokenized_sentences_' + lang1_code + '.p'
if (os.path.exists(lang1_path)):
	tokenized_sentences_lang1 = pd.read_pickle(lang1_path)
else:
	tokenized_sentences_lang1 = get_tokenized_sentences(lang1_code, lang1_name)
	with open(lang1_path, 'wb') as fp:
		pickle.dump(tokenized_sentences_lang1, fp)

lang2_path = WORD_LEVEL_DICTIONARIES_PATH + 'tokenized_sentences_' + lang2_code + '.p'
if (os.path.exists(lang2_path)):
	tokenized_sentences_lang2 = pd.read_pickle(lang2_path)
else:
	tokenized_sentences_lang2 = get_tokenized_sentences(lang2_code, lang2_name)
	with open(lang2_path, 'wb') as fp:
		pickle.dump(tokenized_sentences_lang2, fp)

# Train n gram model
ns = [
	2,
	3,
]
for n in ns:
	print_status('Training word ngrams model... n=' + str(n))
	model_lang1 = NGramModel(n)
	model_lang1.train(tokenized_sentences_lang1)
	write_dict(WORD_LEVEL_DICTIONARIES_PATH, model_lang1.freq_dist, str(n) + '_grams_word_dict_' + lang1_code)

	model_lang2 = NGramModel(n)
	model_lang2.train(tokenized_sentences_lang2)
	write_dict(WORD_LEVEL_DICTIONARIES_PATH, model_lang2.freq_dist, str(n) + '_grams_word_dict_' + lang2_code)

print_status('Done!')