from bs4 import BeautifulSoup
from spacy.lang.en import English
from spacy.lang.es import Spanish
import os
from tools.utils import write_dict
from tools.utils import is_other

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"

def get_frequency_dict(lang):
	frequency_dict = dict()
	other_dict = dict()

	# Load data
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

			nlp = English() if lang is 'en' else Spanish()
			tokenizer = nlp.Defaults.create_tokenizer(nlp)
			tokens = list(tokenizer(cleantext))

			for word in tokens:
				word = word.text.lower()

				if is_other(word):
					if word in other_dict.keys():
						other_dict[word] += 1
					else:
						other_dict[word] = 1
				else:
					if word in frequency_dict.keys():
						frequency_dict[word] += 1
					else:
						frequency_dict[word] = 1

	return frequency_dict, other_dict

def get_probability_dict(frequency_dict):
	nr_of_tokens = sum(frequency_dict.values())
	probability_dict = dict()
	for k, v in frequency_dict.items():
		probability_dict[k] = v / nr_of_tokens
	return probability_dict

# Python code to merge dict using update() method
def merge(dict1, dict2):
	dict2.update(dict1)
	return dict2

# Dictionaries for English
frequency_dict_en, other_dict_en = get_frequency_dict('en')
probability_dict_en = get_probability_dict(frequency_dict_en)
write_dict(WORD_LEVEL_DICTIONARIES_PATH, frequency_dict_en, 'en', probability_dict_en)

# Dictionaries for Spanish
frequency_dict_es, other_dict_es = get_frequency_dict('es')
probability_dict_es = get_probability_dict(frequency_dict_es)
write_dict(WORD_LEVEL_DICTIONARIES_PATH, frequency_dict_es, 'es', probability_dict_es)

# Dictionaries for other class
other_dict = merge(other_dict_en, other_dict_es)
probability_dict_other = get_probability_dict(other_dict)
write_dict(WORD_LEVEL_DICTIONARIES_PATH, other_dict, 'other')

