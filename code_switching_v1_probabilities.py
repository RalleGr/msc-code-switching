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
import os

DICTIONARIES_PATH = "./dictionaries/word-level/"

# Get language codes and evaluation dataset from keyboard
if len(sys.argv) == 1:
	print("Please give two letter language codes as arg, for example en es")
	print("Please enter evaluation dataset: 'dev', 'test' or 'test-original'")
	exit(1)
lang1_code = sys.argv[1]
lang2_code = sys.argv[2]
evaluation_dataset = sys.argv[3]

# Get dictionaries
print_status("Getting dictionaries...")
lang1_path = DICTIONARIES_PATH+'probability_dict_' + lang1_code + '.csv'
lang2_path = DICTIONARIES_PATH+'probability_dict_' + lang2_code + '.csv'
if (os.path.exists(lang1_path) and os.path.exists(lang2_path)):
	probability_lang1_df = pd.read_csv(lang1_path, encoding='utf-16')
	probability_lang1_dict = probability_lang1_df.set_index('word')['probability'].to_dict()

	probability_lang2_df = pd.read_csv(lang2_path, encoding='utf-16')
	probability_lang2_dict = probability_lang2_df.set_index('word')['probability'].to_dict()
else:
	print("Please run: python train_probability.py " + lang1_code + " " + lang2_code)

# Get data
print_status("Getting test data...")
if (evaluation_dataset == 'dev'):
	filepath = './datasets/bilingual-annotated/' + lang1_code + '-' + lang2_code + '/dev.conll' # validation
if (evaluation_dataset == 'test'):
	filepath = './datasets/bilingual-annotated/' + lang1_code + '-' + lang2_code + '/test.conll' # test
if (evaluation_dataset == 'test-original'):
	filepath = './datasets/bilingual-annotated/' + lang1_code + '-' + lang2_code + '/test-original.conll' # original test set from LinCE

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

		# Get lang1 prob
		if word in probability_lang1_dict: prob_lang1 = probability_lang1_dict[word]
		else: prob_lang1 = probability_lang1_dict['OOV']

		# Get lang2 prob
		if word in probability_lang2_dict: prob_lang2 = probability_lang2_dict[word]
		else: prob_lang2 = probability_lang2_dict['OOV']

		# Assign class based on regex or class with highest prob
		if (is_other(word)):
			lang = 'other'
		else:
			if (prob_lang1 >= prob_lang2):
				lang = 'lang1'
			else:
				lang = 'lang2'

		y.append(lang)
		predictions_dict[word] = lang
	else:
		y.append('')

if (evaluation_dataset == 'test-original'):
	save_predictions(y, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_test_original_probabilities.txt')
	exit(1)

# Get accuracy
acc = accuracy_score(t, y)
print("Accuracy: " + str(acc))

# F1 score
f1 = f1_score(t, y, average=None)
print("F1 score per class: " + str(f1))

# F1 score weighted
f1_weighted = f1_score(t, y, average='weighted')
print("Weighted F1 score: " + str(f1_weighted))

# Confusion matrix
conf_matrix = confusion_matrix(t, y)
classes = ['lang1', 'lang2', 'other']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes).plot(values_format='d')
plt.savefig('./results/CM/confusion_matrix_probabilities.svg', format='svg')

# Save model output
if (evaluation_dataset == 'dev'):
	save_predictions(predictions_dict, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_val_probabilities.txt')
if (evaluation_dataset == 'test'):
	save_predictions(predictions_dict, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_test_probabilities.txt')

# Own test set
# Accuracy: 0.8930302566457472
# F1 score per class: [0.88149196 0.85917539 0.9704282 ]
# Weighted F1 score: 0.8927354350501349

###########################################################################
##################### RESULTS FOR WORKSHOP ################################
# Validation (dev) set
# Accuracy: 0.9138921943438743
# F1 score per class: [0.91054322 0.88774562 0.96758294]
# Weighted F1 score: 0.9132189414118914
