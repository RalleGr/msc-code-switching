import json
import os
from tools.utils import printStatus
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

PREDICTIONS_PATH = './results/predictions/'

predictionsFileNames = [
	'predictions_probabilities.txt',
	#'predictions_2_grams.txt',
	#'predictions_3_grams.txt',
	#'predictions_4_grams.txt',
	'predictions_5_grams.txt',
	#'predictions_6_grams.txt',
	#'predictions_word_2_grams.txt',
	'predictions_viterbi_v1.txt',
	#'predictions_viterbi_v2.txt',
	#'predictions_LDA_v1.txt',
	#'predictions_LDA_v2.txt',
	'predictions_SVM.txt',
	'predictions_LogisticRegression.txt',
]

# Read all results
results = []
for file in predictionsFileNames:
	file_path = os.path.join(PREDICTIONS_PATH, file)
	with open(file_path, 'r') as file:
  		results.append(json.load(file))

# Get data
printStatus("Getting test data...")
filepath = 'datasets/bilingual-annotated/dev.conll'
file = open(filepath, 'rt', encoding='utf8')
words = []
t = []
for line in file:
	# Remove empty lines, lines starting with # sent_enum, \n and split on tab
	if (line.strip() is not '' and '# sent_enum' not in line):
		line = line.rstrip('\n')
		splits = line.split("\t")
		if (splits[1] == 'ambiguous' or splits[1] == 'fw' or splits[1] == 'mixed' or splits[1] == 'ne' or splits[1] == 'unk'):
			continue
		else:
			words.append(splits[0].lower())
			t.append(splits[1])
file.close()

nr_words = len(words)
nr_predictions = len(results)

# Create a list of dict elements: {'lang1': nr_of_predictions, 'lang2': nr_of_predictions, 'other': nr_of_predictions}
results_dict = []
labels = ['lang1', 'lang2', 'other'] 
for i in range(nr_words):
	results_dict.append({key: 0 for key in labels})

for i in range(nr_words):
	for j in range(nr_predictions):
		word = words[i]
		predicted_label = results[j][word]
		results_dict[i][predicted_label] += 1

predictions = []
for word in results_dict:
	predictions.append(max(word.items(), key=operator.itemgetter(1))[0])

# Get accuracy
acc = accuracy_score(t, predictions)
print(acc)

# Fq score
f1 = f1_score(t, predictions, average=None)
print(f1)

# Confusion matrix
conf_matrix = confusion_matrix(t, predictions)
classes = ['lang1', 'lang2', 'other']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
					   display_labels=classes).plot(values_format='d')
plt.savefig('./results/CM/confusion_matrix_majority_voting.svg', format='svg')

# Using all outcomes
# 0.9216649365774616
# [0.92103304 0.91323755 0.93879938]

# Using probabilities, 5grams, viterbi_v2, SVM, LogisticRegression
# 0.922677671721903
# [0.92406337 0.91239635 0.93879938]