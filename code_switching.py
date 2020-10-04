import pandas as pd
import os
from sklearn.metrics import accuracy_score

# Get dictionaries
probability_en_df = pd.read_csv('probability_dict_en.csv')
probability_en_dict = probability_en_df.set_index('word')['probability'].to_dict()

probability_es_df = pd.read_csv('probability_dict_es.csv')
probability_es_dict = probability_es_df.set_index('word')['probability'].to_dict()

# Get data
filepath = 'bilingual-annotated/train.conll'
file = open(filepath, 'rt', encoding='utf8')
words = []
t = []
for line in file:
	# Remove empty lines, lines starting with #, \n and split on tab
	if (line.strip() is not '' and line[0] is not '#'):
		line = line.rstrip('\n')
		splits = line.split("\t")
		words.append(splits[0])
		t.append(splits[1])
file.close()

# Choose language with highest probability for each word
y = []
for word in words:
	try:
		prob_en = probability_en_dict[word]
		prob_es = probability_es_dict[word]
		lang = 'lang1' if (prob_en > prob_es) else 'lang2'
		y.append(lang)
	except:
		# TODO need 2 categories, one for punctuation marks and one for unknown words, now both are 'other'
		y.append('other')
		# print(f"{word} doesn't exist in dictionaries")

# Get accuracy
acc = accuracy_score(t, y)
print(acc)
# 0.6959812854571735