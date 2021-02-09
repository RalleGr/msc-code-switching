import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.utils import is_other
from tools.utils import print_status
from tools.utils import merge_dictionaries
from tools.utils import save_predictions
import math
import sys

# Based on own implementation

# Get evaluation dataset from keyboard
if len(sys.argv) == 1:
	print("Please enter evaluation dataset: 'dev', 'test' or 'test-original'")
	exit(1)
evaluation_dataset = sys.argv[1]

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"

def max_argmax(iterable):
	"""Returns the tuple (argmax, max) for a list
	"""
	return max(enumerate(iterable), key=lambda x: x[1])

def viterbi(obs, states, start_p, trans_p, emit_p):
	languages = []

	V = [{}] # Stores max probability for each token and language
	S = [{}] # Stores argmax (most likely language)

	# Initial probabilities for both languages
	for lang in states:
		try: V[0][lang] = math.log(start_p[lang]) + math.log(emit_p[lang][obs[0]])
		except: V[0][lang] = math.log(start_p[lang]) + math.log(0.0001)

	# Iterate over tokens (starting at second token)
	for t in range(1, len(obs)):
		V.append({})
		S.append({})
		# Iterate over the two languages
		for lang in states:
			# Get max and argmax for current position
			term = []
			for lang2 in states:
				if (obs[t] in emit_p[lang]):
					term.append(V[t-1][lang2] + math.log(trans_p[lang2][lang]) + math.log(emit_p[lang][obs[t]]))
				else:
					term.append(V[t-1][lang2] + math.log(trans_p[lang2][lang]) + math.log(0.0001))
			# try:
			# 	term = (V[t-1][lang2] + math.log(trans_p[lang2][lang]) + math.log(emit_p[lang][obs[t]]) for lang2 in states)
			# except:
			# 	term = (V[t-1][lang2] + math.log(trans_p[lang2][lang]) + math.log(0.0001) for lang2 in states)
			maxlang, prob = max_argmax(term)
			V[t][lang] = prob
			S[t][lang] = states[maxlang]

	# Get argmax for final token
	languages = [0] * len(obs)
	languages[-1] = states[max_argmax(V[-1][lang] for lang in states)[0]]

	# Reconstruct optimal path
	for t in range(len(obs)-1, 0, -1):
		languages[t-1] = S[t][languages[t]]

	return languages

# States
states = ['lang1', 'lang2']

# Start probabilities
start_probabilities = {'lang1': 0.5, 'lang2': 0.5}

# Transition probabilities
transition_probabilities = {
	'lang1' : {'lang1': 0.8, 'lang2': 0.2},
	'lang2' : {'lang1': 0.2, 'lang2': 0.8}
}

# Emission probabilities (our probability dictionaries)
print_status("Getting dictionaries...")
probability_en_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH + 'probability_dict_en.csv', encoding='utf-16')
probability_en_dict = probability_en_df.set_index('word')['probability'].to_dict()

probability_es_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH + 'probability_dict_es.csv', encoding='utf-16')
probability_es_dict = probability_es_df.set_index('word')['probability'].to_dict()
print_status("Dictionaries ready!")

data_en = {'lang1': probability_en_dict}
data_es = {'lang2': probability_es_dict}
emission_probabilities = merge_dictionaries(data_en, data_es)

# Get data
print_status("Getting test data...")
if (evaluation_dataset == 'dev'):
	filepath = './datasets/bilingual-annotated/dev.conll' # validation
if (evaluation_dataset == 'test'):
	filepath = './datasets/bilingual-annotated/test.conll' # test
if (evaluation_dataset == 'test-original'):
	filepath = './datasets/bilingual-annotated/test-original.conll' # original test set from LinCE

file = open(filepath, 'rt', encoding='utf8')
sentences = []
t = []
s = []

if (evaluation_dataset != 'test-original'):
	# Own dev/test set
	for line in file:
		# Remove empty lines, lines starting with # sent_enum, \n and split on tab
		if (line.strip() is not ''):
			if ('# sent_enum' in line):
				s = []
			else:
				line = line.rstrip('\n')
				splits = line.split("\t")
				if (splits[1]=='ambiguous' or splits[1]=='fw' or splits[1]=='mixed' or splits[1]=='ne' or splits[1]=='unk'):
					continue
				else:
					s.append(splits[0].lower())
					t.append(splits[1])
		else:
			sentences.append(s)
else:
	# Original test set
	for line in file:
		# Remove empty lines, lines starting with # sent_enum, \n and split on tab
		if (line.strip() is not ''):
			token = line.rstrip('\n')
			s.append(token.lower())
		else:
			sentences.append(s)
			s = []
file.close()

y = []
predictions_dict = dict()
# For each sentence
for tokens in sentences:
	if(len(tokens) > 0):
		# Separate other words from lang1 and lang2 words
		lang_tokens = []
		other_indexes = []
		for i in range(len(tokens)):
			if (is_other(tokens[i])): other_indexes.append(i)
			else: lang_tokens.append(tokens[i])
		if(len(lang_tokens) > 0):
			# Get viterbi state sequence prediction for a sentence
			y_sentence = viterbi(lang_tokens, states, start_probabilities, transition_probabilities, emission_probabilities)
			# Insert 'other' in the sequence at the right indexes
			for index in other_indexes:
				y_sentence.insert(index, 'other')
		# In case a sentence is made up only of 'other' words, append to y_sentence that many times
		else:
			y_sentence = []
			for index in other_indexes:
				y_sentence.append('other')
		for i in range(len(tokens)):
			predictions_dict[tokens[i]] = y_sentence[i]
		y.append(y_sentence)

if (evaluation_dataset == 'test-original'):
	save_predictions(y, './results/predictions/predictions_test_original_viterbi_v2.txt')
	exit(1)

# Flatten y list
y = [item for y_sent in y for item in y_sent]

# Get accuracy
acc = accuracy_score(t, y)
print(acc)

# Fq score
f1 = f1_score(t, y, average=None)
print(f1)

# Confusion matrix
conf_matrix = confusion_matrix(t, y)
classes = ['lang1', 'lang2', 'other']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes).plot(values_format='d')
plt.savefig('./results/CM/confusion_matrix_' + 'viterbi_v2.svg', format='svg')

# Save model output
if (evaluation_dataset == 'dev'):
	save_predictions(predictions_dict, './results/predictions/predictions_val_viterbi_v2.txt')
if (evaluation_dataset == 'test'):
	save_predictions(predictions_dict, './results/predictions/predictions_test_viterbi_v2.txt')

# RESULTS
# Validation set
# 0.8891055016836722
# [0.88475417 0.84983933 0.96758294]

# Test set
# 0.888285116587581
# [0.87916525 0.84769722 0.9704282 ]