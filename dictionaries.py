from bs4 import BeautifulSoup
import spacy
import os
import csv

# MODEL SIZES
## ENGLISH
## en_core_web_sm
## 
## SPANISH
## es_core_news_sm
## es_core_news_md
## es_core_news_lg

def get_frequency_dict(lang, model):
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
			nlp = spacy.load(model)
			nlp.max_length = 1500000
			tokens = nlp(cleantext)

			# remove all tokens that are not alphabetic (fx. “armour-like” and “‘s”)
			# TODO this needs some more fine-grained preprocessing
			# words = [word.lower for word in tokens]
			# print(words[:100])

			for word in tokens:
				if word.is_punct or word.is_digit or word.pos_ == 'SYM':
					if word.text in other_dict.keys():
						other_dict[word.text] += 1
					else:
						other_dict[word.text] = 1
				else:
					if word.text.lower() in frequency_dict.keys():
						frequency_dict[word.text.lower()] += 1
					else:
						frequency_dict[word.text.lower()] = 1

	return frequency_dict, other_dict

def get_probability_dict(frequency_dict):
	nr_of_tokens = sum(frequency_dict.values())
	probability_dict = dict()
	for k, v in frequency_dict.items():
		probability_dict[k] = v / nr_of_tokens
	return probability_dict


def write_dict(frequency_dict, lang, probability_dict=None):
	frequency_dict_csv = csv.writer(open('frequency_dict_' + lang + '.csv', 'w'))
	frequency_dict_csv.writerow(['word', 'frequency'])
	for key, val in frequency_dict.items():
		frequency_dict_csv.writerow([key, val])

	if probability_dict is not None:
		probability_dict_csv = csv.writer(open('probability_dict_' + lang + '.csv', 'w'))
		probability_dict_csv.writerow(['word', 'probability'])
		for key, val in probability_dict.items():
			probability_dict_csv.writerow([key, val])

# Python code to merge dict using update() method
def merge(dict1, dict2):
	dict2.update(dict1)
	return dict2

# Dictionaries for English
frequency_dict_en, other_dict_en = get_frequency_dict('en', 'en_core_web_sm')
probability_dict_en = get_probability_dict(frequency_dict_en)
write_dict(frequency_dict_en, 'en', probability_dict_en)

# Dictionaries for Spanish
frequency_dict_es, other_dict_es = get_frequency_dict('es', 'es_core_news_sm')
probability_dict_es = get_probability_dict(frequency_dict_es)
write_dict(frequency_dict_es, 'es', probability_dict_es)

other_dict = merge(other_dict_en, other_dict_es)
probability_dict_other = get_probability_dict(other_dict)
write_dict(other_dict,'other')

