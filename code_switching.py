import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

DICTIONARIES_PATH = "./dictionaries/word-level/"

# Get dictionaries
probability_en_df = pd.read_csv(DICTIONARIES_PATH+'probability_dict_en.csv', encoding='utf-16')
probability_en_dict = probability_en_df.set_index('word')['probability'].to_dict()

probability_es_df = pd.read_csv(DICTIONARIES_PATH+'probability_dict_es.csv', encoding='utf-16')
probability_es_dict = probability_es_df.set_index('word')['probability'].to_dict()

probability_other_df = pd.read_csv(DICTIONARIES_PATH+'probability_dict_other.csv', encoding='utf-16')
probability_other_dict = probability_other_df.set_index('word')['probability'].to_dict()

# Get data
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

# Choose language with highest probability for each word
y = []
for word in words:
	if word in probability_en_dict:
		prob_en = probability_en_dict[word]
	else:
		prob_en = 0

	if word in probability_es_dict:
		prob_es = probability_es_dict[word]
	else:
		prob_es = 0

	if word in probability_other_dict:
		prob_other = probability_other_dict[word]
	else:
		prob_other = 0


	if(prob_en == 0 and prob_es == 0 and prob_other == 0):
		lang = 'unk'
	else:
		if (prob_en >= prob_es) and (prob_en >= prob_other):
			lang = 'lang1'
		elif (prob_es >= prob_en) and (prob_es >= prob_other):
			lang = 'lang2'
		else:
			lang = 'other'
	y.append(lang)

	"""
	if(prob_en == 0 and prob_es == 0):
		lang = 'other'
	else:
		lang = 'lang1' if (prob_en > prob_es) else 'lang2'
	"""
	"""
	try:
		prob_en = probability_en_dict[word]
		prob_es = probability_es_dict[word]
		lang = 'lang1' if (prob_en > prob_es) else 'lang2'
		y.append(lang)
	except:
		# TODO need 2 categories, one for punctuation marks and one for unknown words, now both are 'other'
		y.append('other')
		# print(f"{word} doesn't exist in dictionaries")
	"""

# Get accuracy
acc = accuracy_score(t, y)
print(acc)
# 0.6959812854571735 # first try
# 0.7643007491479774 # after fixing bug
# 0.7658330075309709 # after implementing tokenizer
# 0.6598583845731594 # after using "other" dictionary

# Fq score
f1 = f1_score(t, y, average=None)
print(f1)

# Confusion matrix
conf_matrix = confusion_matrix(t, y)
classes = ['ambiguous', 'fw', 'lang1', 'lang2', 'mixed', 'ne', 'other', 'unk']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes).plot(values_format='d')
plt.savefig("confusion_matrix.png")

