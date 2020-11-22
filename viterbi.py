# Importing the required libraries 
import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from models.ngrams.word_ngrams import NGramModel
from tools.utils import is_other
from tools.utils import printStatus
from tools.utils import merge_dictionaries
import sys

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"
n = 2

def viterbi(obs, states, start_p, trans_p, emit_p):
	V = [{}]
	path = {}
 
	# Initialize base cases (t == 0)
	for y in states:
		try:
			V[0][y] = start_p[y] * emit_p[y][obs[0]]
		except:
			V[0][y] = start_p[y] * 0.0001
		path[y] = [y]
	 # Run Viterbi for t > 0
	for t in range(1, len(obs)):
		V.append({})
		newpath = {}
 
		for y in states:
			try:
				(prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
			except:
				(prob, state) = max((V[t-1][y0] * trans_p[y0][y] * 0.0001, y0) for y0 in states)
			V[t][y] = prob
			newpath[y] = path[state] + [y]
 
		# Don't need to remember the old paths
		path = newpath
 
	(prob, state) = max((V[t][y], y) for y in states)
	return (prob, path[state])

# States
states = ['lang1', 'lang2']

# Observations
# Get data
filepath = 'datasets/bilingual-annotated/dev.conll'
file = open(filepath, 'rt', encoding='utf8')
observations = []
all_observations = []
t = []
for line in file:
	# Remove empty lines, lines starting with # sent_enum, \n and split on tab
	if (line.strip() is not '' and '# sent_enum' not in line):
		line = line.rstrip('\n')
		splits = line.split("\t")
		if (splits[1]=='ambiguous' or splits[1]=='fw' or splits[1]=='mixed' or splits[1]=='ne' or splits[1]=='unk'):
			continue
		else:
			if (splits[1] != 'other'): observations.append(splits[0].lower())
			all_observations.append(splits[0].lower())
			t.append(splits[1])
file.close()

# Start probabilities
start_probabilities = {'lang1': 0.5, 'lang2': 0.5}

# Transition probabilities
transition_probabilities = {
	'lang1' : {'lang1': 0.8, 'lang2': 0.2},
	'lang2' : {'lang1': 0.2, 'lang2': 0.8}
}

# Emission probabilities (our probability dictionaries)
printStatus("Getting dictionaries...")
probability_en_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH + 'probability_dict_en.csv', encoding='utf-16')
probability_en_dict = probability_en_df.set_index('word')['probability'].to_dict()

probability_es_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH + 'probability_dict_es.csv', encoding='utf-16')
probability_es_dict = probability_es_df.set_index('word')['probability'].to_dict()
printStatus("Dictionaries ready!")

data_en = {'lang1': probability_en_dict}
data_es = {'lang2': probability_es_dict}
emission_probabilities = merge_dictionaries(data_en, data_es)

prob, path = viterbi(observations,
					states,
					start_probabilities,
					transition_probabilities,
					emission_probabilities)

# Choose language
y = []
j = 0
print(path[:100])
print(len(observations))
print(len(all_observations))
for i in range(len(all_observations)):
# for i in range(5):
	word = all_observations[i]
	if (is_other(word)):
		lang = 'other'
	else:
		# print(j)
		try:
			lang = path[j]
			print(lang)
			print(word)
		except:
			lang = 'other'
	y.append(lang)

# Get accuracy
acc = accuracy_score(t, y)
print(acc)
# 0.4984176013368104 # with unigrams and bigrams

# Fq score
f1 = f1_score(t, y, average=None)
print(f1)

# Confusion matrix
conf_matrix = confusion_matrix(t, y)
classes = ['lang1', 'lang2', 'other']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes).plot(values_format='d')
plt.savefig('./results/confusion_matrix_' + 'viterbi.svg',format='svg')