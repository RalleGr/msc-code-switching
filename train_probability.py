from bs4 import BeautifulSoup
from tools.utils import write_dict
from tools.utils import is_other
from tools.utils import print_status
import pandas as pd
import spacy
import os
import importlib
import sys
from langs import langs

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"

def get_frequency_dict(lang_code, lang_name):
	print_status("Creating frequency dictionaries...")

	frequency_dict = dict()

	# Load data
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

			module = importlib.import_module("spacy.lang." + lang_code)
			nlp = getattr(module, lang_name)() if module is not None else spacy.language.Language()
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
	print_status("Creating probability dictionaries...")
	smoothing_factor = 1
	nr_of_tokens = sum(frequency_dict.values())
	nr_of_distinct_words = len(frequency_dict.keys())
	probability_dict = dict()
	for k, v in frequency_dict.items():
		probability_dict[k] = (v + smoothing_factor) / (nr_of_tokens + smoothing_factor * nr_of_distinct_words)
	probability_dict['OOV'] = smoothing_factor / (nr_of_tokens + smoothing_factor * nr_of_distinct_words)
	return probability_dict


# Get language code from keyboard
if len(sys.argv) == 1:
	print("Please give two letter language codes as arg, for example en es")
	print("Get only frequency dictionaries, give frequency arg. Get both frequency and probability, give probability arg")
	exit(1)

# Lang 1
lang1 = sys.argv[1]
lang1_code = langs()[lang1]['code']
lang1_name = langs()[lang1]['name']

# Lang 2
lang2 = sys.argv[2]
lang2_code = langs()[lang2]['code']
lang2_name = langs()[lang2]['name']

# Frequency
fullTraining = sys.argv[3] == 'probability'

# If create frequency dictionaries
frequency_lang1_dict = get_frequency_dict(lang1_code, lang1_name)
frequency_lang2_dict = get_frequency_dict(lang2_code, lang2_name)

# Probability dict
probability_lang1_dict = get_probability_dict(frequency_lang1_dict)
if (fullTraining):
	write_dict(WORD_LEVEL_DICTIONARIES_PATH, frequency_lang1_dict, 'frequency_dict_' + lang1_code, probability_lang1_dict, 'probability_dict_' + lang1_code)
else:
	write_dict(WORD_LEVEL_DICTIONARIES_PATH, frequency_lang1_dict, 'frequency_dict_' + lang1_code)

probability_lang2_dict = get_probability_dict(frequency_lang2_dict)
if (fullTraining):
	write_dict(WORD_LEVEL_DICTIONARIES_PATH, frequency_lang2_dict, 'frequency_dict_' + lang2_code, probability_lang2_dict, 'probability_dict_' + lang2_code)
else:
	write_dict(WORD_LEVEL_DICTIONARIES_PATH, frequency_lang2_dict, 'frequency_dict_' + lang2_code)
print_status('Done!')

