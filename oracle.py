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
	'predictions_val_probabilities.txt',
	'predictions_val_char_5_grams.txt',
	# 'predictions_val_word_2_grams.txt',
	'predictions_val_viterbi_v1.txt',
	'predictions_val_LDA_v2.txt',
	'predictions_val_SVM.txt',
	# 'predictions_val_LogisticRegression.txt',
]

# predictionsFileNames = [
# 	'predictions_test_probabilities.txt',
# 	'predictions_test_char_5_grams.txt',
# 	# 'predictions_test_word_2_grams.txt',
# 	'predictions_test_viterbi_v1.txt',
# 	'predictions_test_LDA_v2.txt',
# 	'predictions_test_SVM.txt',
# 	# 'predictions_test_LogisticRegression.txt',
# ]

# Read all results
results = []
for file in predictionsFileNames:
	file_path = os.path.join(PREDICTIONS_PATH, file)
	with open(file_path, 'r') as file:
  		results.append(json.load(file))

# Get data
printStatus("Getting test data...")
filepath = 'datasets/bilingual-annotated/dev.conll' # validation
# filepath = 'datasets/bilingual-annotated/train.conll' # test
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

predictions = []
for i in range(nr_words):
	predicted_label = ""
	for j in range(nr_predictions):
		word = words[i]
		predicted_label = results[j][word]
		if(predicted_label == t[i]):
			break
	predictions.append(predicted_label)

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
plt.savefig('./results/CM/confusion_matrix_oracle.svg', format='svg')

# RESULTS

# Validation set

# All outcomes
# 0.9851887485125452
# [0.99263341 0.98619647 0.96758294]

# Using probabilities, 5grams, viterbi_v2, SVM, LogisticRegression
# 0.9791882927817303
# [0.98551594 0.97827108 0.96758294]

# Using probabilities, 5grams, viterbi_v2, LDA, SVM
# 0.9825303187583867
# [0.98703859 0.98541351 0.96758294]

# Test set

# All outcomes
# 0.9857390683198123
# [0.99275907 0.98663358 0.9704282 ]

# Using probabilities, 5grams, viterbi_v2, SVM, LogisticRegression - better than majority voting
# 0.969717842747079
# [0.97358582 0.96484662 0.9704282 ]

# Using probabilities, 5grams, viterbi_v2, LDA, SVM - better than majority voting
# 0.9815041583754274
# [0.98084909 0.98897098 0.9704282 ]

