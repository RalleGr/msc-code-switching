import json
import os
from tools.utils import printStatus
from tools.utils import is_other
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import itertools

PREDICTIONS_PATH = './results/predictions/'

# predictionsFileNames = [
# 	'predictions_val_probabilities.txt',
# 	'predictions_val_char_5_grams.txt',
# 	# 'predictions_val_word_2_grams.txt',
# 	'predictions_val_viterbi_v1.txt',
# 	'predictions_val_LDA_v2.txt',
# 	# 'predictions_val_SVM.txt',
# 	'predictions_val_LogisticRegression.txt',
# ]

predictionsFileNames = [
	'predictions_test_probabilities.txt',
	'predictions_test_char_5_grams.txt',
	# 'predictions_test_word_2_grams.txt',
	'predictions_test_viterbi_v1.txt',
	'predictions_test_LDA_v2.txt',
	'predictions_test_SVM.txt',
	# 'predictions_test_LogisticRegression.txt',
]

# perms = list(itertools.permutations(predictionsFileNames, r=5))
perms = [predictionsFileNames]

# Get data
printStatus("Getting test data...")
# filepath = 'datasets/bilingual-annotated/dev.conll' # validation
filepath = 'datasets/bilingual-annotated/test.conll' # test
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

best_acc = 0
best_predictions = None
best_ensembly = None
for perm in perms:
	# Read all results
	results = []
	for file in perm:
		file_path = os.path.join(PREDICTIONS_PATH, file)
		with open(file_path, 'r') as file:
			results.append(json.load(file))

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
	# print(acc)
	if (acc > best_acc):
		best_acc = acc
		best_predictions = predictions
		best_ensembly = perm

print(best_ensembly)
print(best_acc)

# Fq score
f1 = f1_score(t, best_predictions, average=None)
print(f1)

# Confusion matrix
conf_matrix = confusion_matrix(t, best_predictions)
classes = ['lang1', 'lang2', 'other']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
					   display_labels=classes).plot(values_format='d')
plt.savefig('./results/CM/confusion_matrix_majority_voting.svg', format='svg')

# RESULTS

# Validation set

# All outcomes
# 0.9274881636579994
# [0.92509976 0.90803219 0.96758294]

# Best 3 models
# ('predictions_val_char_5_grams.txt', 'predictions_val_viterbi_v1.txt', 'predictions_val_SVM.txt')
# 0.9373876496949135
# [0.93546508 0.9231036  0.96758294]

# Best 5 models
# ('predictions_val_probabilities.txt', 'predictions_val_char_5_grams.txt', 'predictions_val_viterbi_v1.txt', 'predictions_val_SVM.txt', 'predictions_val_LogisticRegression.txt')
# 0.934779856697977
# [0.93283172 0.9189907  0.96758294]

# not best, but close enough and with more different implementations
# ['predictions_val_probabilities.txt', 'predictions_val_char_5_grams.txt', 'predictions_val_viterbi_v1.txt', 'predictions_val_LDA_v2.txt', 'predictions_val_SVM.txt']
# 0.9301972301693799
# [0.92787963 0.91224497 0.96758294]

# Test set

# All outcomes
# 0.9062197050869942
# [0.89455143 0.88112602 0.9704282 ]

# Best 3 models
# ('predictions_test_viterbi_v1.txt', 'predictions_test_SVM.txt', 'predictions_test_LogisticRegression.txt')
# 0.9174957906015613
# [0.90606911 0.89919211 0.9704282 ]

# Best 5 models
# ('predictions_test_probabilities.txt', 'predictions_test_char_5_grams.txt', 'predictions_test_viterbi_v1.txt', 'predictions_test_SVM.txt', 'predictions_test_LogisticRegression.txt')
# 0.9152507781009235
# [0.90410553 0.8951825  0.9704282 ]

# not best, but close enough and with more different implementations
# ['predictions_test_probabilities.txt', 'predictions_test_char_5_grams.txt', 'predictions_test_viterbi_v1.txt', 'predictions_test_LDA_v2.txt', 'predictions_test_SVM.txt']
# 0.9067809582121537
# [0.89515916 0.88199527 0.9704282 ]

