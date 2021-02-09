import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.utils import save_predictions
from tools.utils import is_other
from tools.utils import print_status
import sys

DICTIONARIES_PATH = "./dictionaries/word-level/"

# Get evaluation dataset from keyboard
if len(sys.argv) == 1:
	print("Please enter evaluation dataset: 'dev', 'test' or 'test-original'")
	exit(1)
evaluation_dataset = sys.argv[1]

# Get dictionaries
print_status("Getting dictionaries...")
probability_en_df = pd.read_csv(DICTIONARIES_PATH+'probability_dict_en.csv', encoding='utf-16')
probability_en_dict = probability_en_df.set_index('word')['probability'].to_dict()

probability_es_df = pd.read_csv(DICTIONARIES_PATH+'probability_dict_es.csv', encoding='utf-16')
probability_es_dict = probability_es_df.set_index('word')['probability'].to_dict()

# Get data
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
			if (splits[1]=='ambiguous' or splits[1]=='fw' or splits[1]=='mixed' or splits[1]=='ne' or splits[1]=='unk'):
				continue
			else:
				words.append(splits[0])
				t.append(splits[1])
else:
	# Original test set
	for line in file:
		# Remove empty lines, lines starting with # sent_enum, \n and split on tab
		if (line.strip() is not ''):
			token = line.rstrip('\n')
			words.append(token.lower())
		else:
			words.append('')

file.close()

# Choose language with highest probability for each word
y = []
predictions_dict = dict()
for word in words:
	if (word != ''):
		word = word.lower()

		# Get EN prob
		if word in probability_en_dict: prob_en = probability_en_dict[word]
		else: prob_en = probability_en_dict['OOV']

		# Get ES prob
		if word in probability_es_dict: prob_es = probability_es_dict[word]
		else: prob_es = probability_es_dict['OOV']

		# Assign class based on regex or class with highest prob
		if (is_other(word)):
			lang = 'other'
		else:
			if (prob_en >= prob_es):
				lang = 'lang1'
			else:
				lang = 'lang2'

		y.append(lang)
		predictions_dict[word] = lang
	else:
		y.append('')

if (evaluation_dataset == 'test-original'):
	save_predictions(y, './results/predictions/predictions_test_original_probabilities.txt')
	exit(1)

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
plt.savefig('./results/CM/confusion_matrix_probabilities.svg', format='svg')

# Save model output
if (evaluation_dataset == 'dev'):
	save_predictions(predictions_dict, './results/predictions/predictions_val_probabilities.txt')
if (evaluation_dataset == 'test'):
	save_predictions(predictions_dict, './results/predictions/predictions_test_probabilities.txt')

# RESULTS
# Validation (dev) set
# 0.9138921943438743
# [0.91054322 0.88774562 0.96758294]

# Test set
# 0.8930302566457472
# [0.88149196 0.85917539 0.9704282 ]