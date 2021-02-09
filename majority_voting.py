import json
import os
from tools.utils import print_status
from tools.utils import is_other
from tools.utils import save_predictions
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import itertools
import sys

PREDICTIONS_PATH = './results/predictions/'

# Get evaluation dataset from keyboard
if len(sys.argv) == 1:
	print("Please enter evaluation dataset: 'dev', 'test' or 'test-original'")
	exit(1)
evaluation_dataset = sys.argv[1]

# Get test data
print_status("Getting predictions data...")
if (evaluation_dataset == 'dev'):
	predictionsFileNames = [
		'predictions_val_probabilities.txt',
		'predictions_val_char_5_grams.txt',
		# 'predictions_val_word_2_grams.txt',
		'predictions_val_viterbi_v1.txt',
		'predictions_val_LDA.txt',
		'predictions_val_SVM.txt',
		# 'predictions_val_LogisticRegression.txt',
	]
if (evaluation_dataset == 'test'):
	predictionsFileNames = [
		'predictions_test_probabilities.txt',
		'predictions_test_char_5_grams.txt',
		'predictions_test_word_2_grams.txt',
		'predictions_test_viterbi_v1.txt',
		'predictions_test_LDA.txt',
		'predictions_test_SVM.txt',
		'predictions_test_LogisticRegression.txt',
	]
if (evaluation_dataset == 'test-original'):
	predictionsFileNames = [
		'predictions_test_original_probabilities.txt',
		'predictions_test_original_char_5_grams.txt',
		'predictions_test_original_word_2_grams.txt',
		'predictions_test_original_viterbi_v1.txt',
		'predictions_test_original_LDA.txt',
		'predictions_test_original_SVM.txt',
		'predictions_test_original_LogisticRegression.txt',
	]

# perms = list(itertools.permutations(predictionsFileNames, r=5))
perms = [predictionsFileNames]

# Get test data
print_status("Getting test data...")
if (evaluation_dataset == 'dev'):
	filepath = './datasets/bilingual-annotated/dev.conll' # validation
if (evaluation_dataset == 'test'):
	filepath = './datasets/bilingual-annotated/test.conll' # test
if (evaluation_dataset == 'test-original'):
	filepath = './datasets/bilingual-annotated/test-original.conll' # original test set from LinCE

file = open(filepath, 'rt', encoding='utf8')
words = []
t = []

if (evaluation_dataset != 'test-original'):
	# Own dev/test set
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
else:
	# Original test set
	for line in file:
		token = line.rstrip('\n')
		words.append(token.lower())
file.close()

best_acc = 0
best_predictions = None
best_ensembly = None
for perm in perms:
	# Read all results
	results = []

	if (evaluation_dataset != 'test-original'):
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
	else:
		for file in perm:
			result = []
			file_path = os.path.join(PREDICTIONS_PATH, file)
			with open(file_path, 'r') as file:
				for line in file:
					pred = line.rstrip('\n')
					result.append(pred)
			results.append(result)
		
		nr_words = len(words)
		nr_predictions = len(results)

		# Create a list of dict elements: {'lang1': nr_of_predictions, 'lang2': nr_of_predictions, 'other': nr_of_predictions}
		results_dict = []
		labels = ['lang1', 'lang2', 'other'] 
		for i in range(nr_words):
			results_dict.append({key: 0 for key in labels})

		predictions = []
		for i in range(nr_words):
			if (words[i] != ''):
				for j in range(nr_predictions):
					predicted_label = results[j][i]
					results_dict[i][predicted_label] += 1
				predictions.append(max(results_dict[i].items(), key=operator.itemgetter(1))[0])
			else:
				predictions.append('')
		
		best_ensembly = perm
		best_predictions = predictions

print(best_ensembly)

if (evaluation_dataset == 'test-original'):
	save_predictions(best_predictions, './results/predictions/predictions_test_original_ensemble_all.txt')
	exit(1)

# Get accuracy
print("Accuracy: " + str(best_acc))

# F1 score
f1 = f1_score(t, best_predictions, average=None)
print("F1 score per class: " + str(f1))

# F1 score weighted
f1_weighted = f1_score(t, best_predictions, average='weighted')
print("Weighted F1 score: " + str(f1_weighted))

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
# ['predictions_val_probabilities.txt', 'predictions_val_char_5_grams.txt', 'predictions_val_viterbi_v1.txt', 'predictions_val_LDA.txt', 'predictions_val_SVM.txt']
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
# ['predictions_test_probabilities.txt', 'predictions_test_char_5_grams.txt', 'predictions_test_viterbi_v1.txt', 'predictions_test_LDA.txt', 'predictions_test_SVM.txt']
# 0.9067809582121537
# [0.89515916 0.88199527 0.9704282 ]


###########################################################################
##################### RESULTS FOR WORKSHOP ################################
# Dev set

# All models
# ['predictions_val_probabilities.txt', 'predictions_val_char_5_grams.txt', 'predictions_val_word_2_grams.txt', 'predictions_val_viterbi_v1.txt', 'predictions_val_LDA.txt', 'predictions_val_SVM.txt', 'predictions_val_LogisticRegression.txt']
# Accuracy: 0.9220447122566271
# F1 score per class: [0.91916767 0.90004613 0.96758294]
# Weighted F1 score: 0.9215255409360907

# Models 2, 4 and 6
# ['predictions_val_char_5_grams.txt', 'predictions_val_viterbi_v1.txt', 'predictions_val_SVM.txt']
# Accuracy: 0.9373876496949135
# F1 score per class: [0.93546508 0.9231036  0.96758294]
# Weighted F1 score: 0.937151714285826

# Models 1, 2, 4, 5 and 6
# ['predictions_val_probabilities.txt', 'predictions_val_char_5_grams.txt', 'predictions_val_viterbi_v1.txt', 'predictions_val_LDA.txt', 'predictions_val_SVM.txt']
# Accuracy: 0.9300200015191027
# F1 score per class: [0.9278748  0.91172122 0.96758294]
# Weighted F1 score: 0.9296303238372436