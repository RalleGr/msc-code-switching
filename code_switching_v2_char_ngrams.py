import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from models.ngrams.ngrams import NGramModel
from tools.utils import save_predictions
from tools.utils import is_other
from tools.utils import print_status
import sys
import os

CHAR_LEVEL_DICTIONARIES_PATH = "./dictionaries/character-level/"

# Get language codes, evaluation dataset and n from keyboard
if len(sys.argv) == 1:
	print("Please give two letter language codes as arg, for example en es")
	print("Please enter evaluation dataset: 'dev', 'test' or 'test-original'")
	print("Please enter n value")
	exit(1)
lang1_code = sys.argv[1]
lang2_code = sys.argv[2]
evaluation_dataset = sys.argv[3]
n = int(sys.argv[4])

if n!=2 and n!=3 and n!=4 and n!=5 and n!=6:
	print("n should be 2, 3, 4, 5 or 6")
	exit(1)

# Get dictionaries
print_status("Getting dictionaries...")
lang1_path = CHAR_LEVEL_DICTIONARIES_PATH + str(n) + '_grams_dict_' + lang1_code + '.csv'
lang2_path = CHAR_LEVEL_DICTIONARIES_PATH + str(n) + '_grams_dict_' + lang2_code + '.csv'
if (os.path.exists(lang1_path) and os.path.exists(lang2_path)):
	frequency_lang1_df = pd.read_csv(lang1_path, encoding='utf-16', converters={"word": ast.literal_eval})
	frequency_lang1_dict = frequency_lang1_df.set_index('word')['frequency'].to_dict()

	frequency_lang2_df = pd.read_csv(lang2_path, encoding='utf-16', converters={"word": ast.literal_eval})
	frequency_lang2_dict = frequency_lang2_df.set_index('word')['frequency'].to_dict()
else:
	print("Please run: python train_ngrams_character.py " + lang1_code + " " + lang2_code + " " + n)

# Apply ngram model to en, es and other
print_status("Applying NGRAM model...")
model_lang1 = NGramModel(n)
model_lang1.load_ngrams_freq(frequency_lang1_dict)

model_lang2 = NGramModel(n)
model_lang2.load_ngrams_freq(frequency_lang2_dict)

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

# Choose language with highest probability for each word based on ngrams
y = []
predictions_dict = dict()
counter = 0
print_status("Classifying...")
for word in words:
	if (word != ''):
		word = word.lower()
		if is_other(word):
			lang = 'other'
		else:
			prob_lang1 = model_lang1.get_word_log_prob(word)
			prob_lang2 = model_lang2.get_word_log_prob(word)
			if (prob_lang1 >= prob_lang2):
				lang = 'lang1'
			else:
				lang = 'lang2'
		
		y.append(lang)
		predictions_dict[word] = lang

		if counter % 10000 == 0:
			print(f"{counter} of {len(words)}")
		counter+=1
	else:
		y.append('')

if (evaluation_dataset == 'test-original'):
	save_predictions(y, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_test_original_char_' + str(n) + '_grams.txt')
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
plt.savefig('./results/CM/confusion_matrix_char_' + str(n) + '_grams.svg', format='svg')

# Save model output
if (evaluation_dataset == 'dev'):
	save_predictions(predictions_dict, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_val_char_' + str(n) + '_grams.txt')
if (evaluation_dataset == 'test'):
	save_predictions(predictions_dict, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_test_char_' + str(n) + '_grams.txt')


# RESULTS
# Validation set
# n = 2
# 0.8066182241689243
# [0.79448622 0.72915367 0.96758294]
# n = 3
# 0.8886244524900625
# [0.87813421 0.85783466 0.96758294]
# n = 4
# 0.9112337645897157
# [0.90487919 0.88800963 0.96758294]
# n = 5
# 0.9222725776641264
# [0.91839397 0.90193206 0.96758294]
# n = 6
# 0.920449654404132
# [0.91676281 0.89876127 0.96758294]

# Own test set
# n = 2
# 0.7999387723863463
# [0.77888372 0.71878903 0.9704282 ]
# n = 3
# 0.8839736721261289
# [0.8669658  0.85202084 0.9704282 ]
# n = 4
# 0.8843818562171539
# [0.8663137  0.85413864 0.9704282 ]
# n = 5
# 0.8984897188632073
# [0.88468719 0.87143629 0.9704282 ]
# n = 6
# 0.8992550640338792
# [0.88685714 0.87073786 0.9704282 ]

###########################################################################
##################### RESULTS FOR WORKSHOP ################################
# Dev set
# n = 5
# Accuracy: 0.9222725776641264
# F1 score per class: [0.91839397 0.90193206 0.96758294]
# Weighted F1 score: 0.9219122579450858
