import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from ngrams.ngrams import NGramModel
from tools.utils import write_dict

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"
CHAR_LEVEL_DICTIONARIES_PATH = "./dictionaries/character-level/"

# Get dictionaries
print("Getting dictionaries...")
frequency_en_df = pd.read_csv(CHAR_LEVEL_DICTIONARIES_PATH+'frequency_dict_en.csv',encoding='utf-16')
frequency_en_dict = frequency_en_df.set_index('word')['frequency'].to_dict()

frequency_es_df = pd.read_csv(CHAR_LEVEL_DICTIONARIES_PATH+'frequency_dict_es.csv',encoding='utf-16')
frequency_es_dict = frequency_es_df.set_index('word')['frequency'].to_dict()

frequency_other_df = pd.read_csv(CHAR_LEVEL_DICTIONARIES_PATH+'frequency_dict_other.csv',encoding='utf-16')
frequency_other_dict = frequency_other_df.set_index('word')['frequency'].to_dict()

# Apply ngram model to en, es and other
print("Applying NGRAM model...")
model_en = NGramModel()
model_en.load_ngrams_freq(frequency_en_dict)

model_es = NGramModel()
model_es.load_ngrams_freq(frequency_es_dict)

model_other = NGramModel()
model_other.load_ngrams_freq(frequency_other_dict)


# Get data
print("Getting test data...")
filepath = 'datasets/bilingual-annotated/test.conll'
file = open(filepath, 'rt', encoding='utf8')
words = []
t = []
for line in file:
	# Remove empty lines, lines starting with # sent_enum, \n and split on tab
	if (line.strip() is not '' and '# sent_enum' not in line):
		line = line.rstrip('\n')
		splits = line.split("\t")
		words.append(splits[0])
		t.append(splits[1])
file.close()

# Choose language with highest probability for each word based on ngrams
y = []
counter = 0
print("Classifying...")
for word in words:
	prob_en = model_en.get_word_log_prob(word)
	prob_es = model_es.get_word_log_prob(word)
	prob_other = model_other.get_word_log_prob(word)
	if (prob_en >= prob_es) and (prob_en >= prob_other):
		lang = 'lang1'
	elif (prob_es >= prob_en) and (prob_es >= prob_other):
		lang = 'lang2'
	else:
		lang = 'other'
	if counter % 1000 == 0:
		print(f"{counter} of {len(words)}")
	y.append(lang)
	counter+=1

# Get accuracy
acc = accuracy_score(t, y)
print(acc)
# 0.3306321355653757 # first try with unigrams and bigrams

# Fq score
f1 = f1_score(t, y, average=None)
print(f1)

# Confusion matrix
conf_matrix = confusion_matrix(t, y)
classes = ['ambiguous', 'fw', 'lang1', 'lang2', 'mixed', 'ne', 'other', 'unk']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes).plot(values_format='d')
plt.savefig("confusion_matrix_ngrams.png")