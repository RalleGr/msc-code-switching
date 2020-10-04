from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import os
import csv

def get_frequency_dict(lang):
	frequency_dict = dict()
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

			tokens = word_tokenize(cleantext)
			# remove all tokens that are not alphabetic (fx. “armour-like” and “‘s”)
			# TODO this needs some more fine-grained preprocessing
			words = [word.lower() for word in tokens if word.isalpha()]
			# print(words[:100])

			for word in words:
				if word in frequency_dict.keys():
					frequency_dict[word] += 1
				else:
					frequency_dict[word] = 1
	return frequency_dict

def get_probability_dict(frequency_dict):
	nr_of_tokens = sum(frequency_dict.values())
	probability_dict = dict()
	for k, v in frequency_dict.items():
		probability_dict[k] = v / nr_of_tokens
	return probability_dict


def write_dict(frequency_dict, probability_dict, lang):
	frequency_dict_csv = csv.writer(open('frequency_dict_' + lang + '.csv', 'w'))
	probability_dict_csv = csv.writer(open('probability_dict_' + lang + '.csv', 'w'))
	frequency_dict_csv.writerow(['word', 'frequency'])
	for key, val in frequency_dict.items():
		frequency_dict_csv.writerow([key, val])
	probability_dict_csv.writerow(['word', 'probability'])
	for key, val in probability_dict.items():
		probability_dict_csv.writerow([key, val])

# Dictionaries for English
frequency_dict_en = get_frequency_dict('en')
probability_dict_en = get_probability_dict(frequency_dict_en)
write_dict(frequency_dict_en, probability_dict_en, 'en')

# Dictionaries for Spanish
frequency_dict_es = get_frequency_dict('es')
probability_dict_es = get_probability_dict(frequency_dict_es)
write_dict(frequency_dict_es, probability_dict_es, 'es')
