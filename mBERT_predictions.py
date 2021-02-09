from tools.utils import print_status
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.utils import save_predictions
import sys

PREDICTIONS_PATH = './results/predictions/'

# Get evaluation dataset from keyboard
if len(sys.argv) == 1:
	print("Please enter evaluation dataset: 'dev', 'test' or 'test-original'")
	exit(1)
evaluation_dataset = sys.argv[1]

# Get predictions data
print_status("Getting predictions data...")
if (evaluation_dataset == 'dev'):
	predictionsFileName = PREDICTIONS_PATH + 'mBERT_predictions_dev.out' # validation
if (evaluation_dataset == 'test'):
	predictionsFileName = PREDICTIONS_PATH + 'mBERT_predictions_test.out' # test

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


# Get test data
print_status("Getting test data...")
if (evaluation_dataset == 'dev'):
	filepath = './datasets/bilingual-annotated/dev.conll' # validation
if (evaluation_dataset == 'test'):
	filepath = './datasets/bilingual-annotated/test.conll' # test
if (evaluation_dataset == 'test-original'):
	filepath = './datasets/bilingual-annotated/test-original.conll' # original test set from LinCE

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

if (evaluation_dataset == 'test-original'):
	save_predictions(y_, './results/predictions/predictions_test_original_mBERT.txt')
	exit(1)
# Get accuracy
acc = accuracy_score(t_, y_)
print("Accuracy: " + str(acc))

# F1 score
f1 = f1_score(t_, y_, average=None)
print("F1 score per class: " + str(f1))

# F1 score weighted
f1_weighted = f1_score(t_, y_, average='weighted')
print("Weighted F1 score: " + str(f1_weighted))

# Confusion matrix
conf_matrix = confusion_matrix(t_, y_)
classes = ['ambiguous', 'fw', 'lang1', 'lang2', 'mixed', 'ne', 'other', 'unk']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
					   display_labels=classes).plot(values_format='d')
plt.savefig('./results/CM/mBERT_test.svg', format='svg')

# RESULTS

# Validation set
# 0.9910626123503051
# [0.   0.   0.99140693   0.99081154   0.   0.99782942   0.]

# Test set
# 0.9859431603653248
# [0.   0.98548717   0.98404558   0.   0.99764318   0.]

###########################################################################
##################### RESULTS FOR WORKSHOP ################################
# Dev set
# Accuracy: 0.9910626123503051
# F1 score per class: [0.         0.         0.99140693 0.99081154 0.         0.99782942
#  0.        ]
# Weighted F1 score: 0.992454704075842