from bs4 import BeautifulSoup
from spacy.lang.en import English
from spacy.lang.es import Spanish
import os
from tools.utils import write_dict
from tools.utils import is_other
import pandas as pd

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"

def get_frequency_dict(lang):
	frequency_dict = dict()

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
					continue
				else:
					if word in frequency_dict.keys():
						frequency_dict[word] += 1
					else:
						frequency_dict[word] = 1

	return frequency_dict

def get_probability_dict(frequency_dict):
	smoothing_factor = 1
	nr_of_tokens = sum(frequency_dict.values())
	nr_of_distinct_words = len(frequency_dict.keys())
	probability_dict = dict()
	for k, v in frequency_dict.items():
		probability_dict[k] = (v + smoothing_factor) / (nr_of_tokens + smoothing_factor * nr_of_distinct_words)
	probability_dict['OOV'] = smoothing_factor / (nr_of_tokens + smoothing_factor * nr_of_distinct_words)
	return probability_dict

# EN

# Uncomment this to create a frequency dict from monolingual datasets
# frequency_en_dict = get_frequency_dict('en')

# Uncomment this to get existing frequency dict
frequency_en_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH + 'frequency_dict_en.csv', encoding='utf-16')
frequency_en_dict = frequency_en_df.set_index('word')['frequency'].to_dict()

# Probability dict
probability_dict_en = get_probability_dict(frequency_en_dict)
write_dict(WORD_LEVEL_DICTIONARIES_PATH, frequency_en_dict, 'frequency_dict_en', probability_dict_en, 'probability_dict_en')

# ES
# Uncomment this to get frequency dict from monolingual datasets
# frequency_es_dict = get_frequency_dict('es')

# Uncomment this to get existing frequency dict
frequency_es_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH + 'frequency_dict_es.csv', encoding='utf-16')
frequency_es_dict = frequency_es_df.set_index('word')['frequency'].to_dict()

# Probability dict
probability_dict_es = get_probability_dict(frequency_es_dict)
write_dict(WORD_LEVEL_DICTIONARIES_PATH, frequency_es_dict, 'frequency_dict_es', probability_dict_es, 'probability_dict_es')


