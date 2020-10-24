from bs4 import BeautifulSoup
import spacy
from spacy.lang.en import English
from spacy.lang.es import Spanish
import os
import emoji
import csv
import string

# MODEL SIZES
## ENGLISH
## en_core_web_sm
## 
## SPANISH
## es_core_news_sm
## es_core_news_md
## es_core_news_lg

DICTIONARIES_PATH = "./dictionaries/"

# Punctuation, Numbers and Emojis
def is_other(word):
	def isfloat(value):
		try:
			float(value)
			return True
		except ValueError:
			return False

	if word in emoji.UNICODE_EMOJI or word.isnumeric() or isfloat(word):
		return True

	for c in word:
		if c not in string.punctuation:
			return False
	return True


def get_frequency_dict(lang):
	frequency_dict = dict()
	other_dict = dict()

	# Load data
	for root, dirs, files in os.walk('monolingual-' + lang):
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

			# tokens = word_tokenize(cleantext)

			# nlp = spacy.load(model)
			# nlp.max_length = 1500000
			# tokens = nlp(cleantext)

			nlp = English() if lang is 'en' else Spanish()
			tokenizer = nlp.Defaults.create_tokenizer(nlp)
			tokens = list(tokenizer(cleantext))

			# print(True if "don't" in (w.text for w in s) else False)


			# remove all tokens that are not alphabetic (fx. “armour-like” and “‘s”)
			# TODO this needs some more fine-grained preprocessing
			# words = [word.lower for word in tokens]
			# print(words[:100])

			for word in tokens:
				word = word.text.lower()
				if is_other(word):
				# if word.is_punct or word.is_digit or word.pos_ == 'SYM':
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


def write_dict(frequency_dict, lang, probability_dict=None):
	frequency_dict_csv = csv.writer(open(DICTIONARIES_PATH+'frequency_dict_' + lang + '.csv', 'w',encoding='UTF-16'))
	frequency_dict_csv.writerow(['word', 'frequency'])
	for key, val in frequency_dict.items():
		frequency_dict_csv.writerow([key, val])

	if probability_dict is not None:
		probability_dict_csv = csv.writer(open(DICTIONARIES_PATH +'probability_dict_' + lang + '.csv', 'w',encoding='UTF-16'))
		probability_dict_csv.writerow(['word', 'probability'])
		for key, val in probability_dict.items():
			probability_dict_csv.writerow([key, val])

# Python code to merge dict using update() method
def merge(dict1, dict2):
	dict2.update(dict1)
	return dict2

# Dictionaries for English
frequency_dict_en, other_dict_en = get_frequency_dict('en')
probability_dict_en = get_probability_dict(frequency_dict_en)
write_dict(frequency_dict_en, 'en', probability_dict_en)

# Dictionaries for Spanish
frequency_dict_es, other_dict_es = get_frequency_dict('es')
probability_dict_es = get_probability_dict(frequency_dict_es)
write_dict(frequency_dict_es, 'es', probability_dict_es)

other_dict = merge(other_dict_en, other_dict_es)
probability_dict_other = get_probability_dict(other_dict)
write_dict(other_dict,'other')

