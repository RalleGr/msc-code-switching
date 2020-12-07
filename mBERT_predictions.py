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

predictionsFileName = PREDICTIONS_PATH + 'mBERT_predictions_dev.out'
# predictionsFileName = 'mBERT_predictions_train.out' # our test dataset

# Get predictions
file = open(predictionsFileName, 'rt', encoding='utf8')
y = []
for line in file:
	# Remove empty lines, lines starting with # sent_enum, \n and split on tab
	if (line.strip() is not '' and '# sent_enum' not in line):
		line = line.rstrip('\n')
		splits = line.split("\t")
		pred = splits[1]
		y.append(pred)
file.close()


# Get annotated data
printStatus("Getting test data...")
filepath = 'datasets/bilingual-annotated/dev.conll' # validation
# filepath = 'datasets/bilingual-annotated/train.conll' # test
file = open(filepath, 'rt', encoding='utf8')
t = []
for line in file:
	# Remove empty lines, lines starting with # sent_enum, \n and split on tab
	if (line.strip() is not '' and '# sent_enum' not in line):
		line = line.rstrip('\n')
		splits = line.split("\t")
		label = splits[1]
		t.append(label)
file.close()

t_ = []
y_ = []

for i in range(len(t)):
	label = t[i]
	pred = y[i]
	if (label == 'ambiguous' or label == 'fw' or label == 'mixed' or label == 'ne' or label == 'unk'):
		continue
	else:
		t_.append(label)
		y_.append(pred)

# Accuracy
acc = accuracy_score(t_, y_)
print(acc)

# Fq score
f1 = f1_score(t_, y_, average=None)
print(f1)

# Confusion matrix
conf_matrix = confusion_matrix(t_, y_)
classes = ['lang1', 'lang2', 'other']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
					   display_labels=classes).plot(values_format='d')
plt.savefig('./results/CM/mBERT.svg', format='svg')

# RESULTS

# Validation set
# 0.9910626123503051
# [0.   0.   0.99140693   0.99081154   0.   0.99782942   0.]

# Test set

