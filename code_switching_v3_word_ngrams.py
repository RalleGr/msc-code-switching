import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from models.ngrams.word_ngrams import NGramModel
from tools.utils import is_other
from tools.utils import print_status
from tools.utils import save_predictions
import sys
import os

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"

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
if n!=2 and n!=3:
	print("n should be 2 or 3")
	exit(1)

# Get dictionaries
print_status("Getting dictionaries...")
lang1_path = WORD_LEVEL_DICTIONARIES_PATH + str(n) + '_grams_word_dict_' + lang1_code + '.csv'
lang2_path = WORD_LEVEL_DICTIONARIES_PATH + str(n) + '_grams_word_dict_' + lang2_code + '.csv'
if (os.path.exists(lang1_path) and os.path.exists(lang2_path)):
	frequency_lang1_df = pd.read_csv(lang1_path, encoding='utf-16', converters={"word": ast.literal_eval})
	frequency_lang1_dict = frequency_lang1_df.set_index('word')['frequency'].to_dict()

	frequency_lang2_df = pd.read_csv(lang2_path, encoding='utf-16', converters={"word": ast.literal_eval})
	frequency_lang2_dict = frequency_lang2_df.set_index('word')['frequency'].to_dict()
else:
	print("Please run: python train_ngrams_word.py " + lang1_code + " " + lang2_code + " " + n)

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
sentences = []
t = []
s = []
if (evaluation_dataset != 'test-original'):
	# Own dev/test set
	for line in file:
		# Remove empty lines, lines starting with # sent_enum, \n and split on tab
		if (len(line.strip()) != 0):
			if ('# sent_enum' in line):
				sentences.append(s)
				s = []
			else:
				line = line.rstrip('\n')
				splits = line.split("\t")
				if (splits[1]=='ambiguous' or splits[1]=='fw' or splits[1]=='mixed' or splits[1]=='ne' or splits[1]=='unk'):
					continue
				else:
					s.append(splits[0].lower())
					t.append(splits[1])
	sentences.append(s) # append last sentence
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

# Choose language with highest probability for each word based on ngrams
y = []
predictions_dict = dict()
counter = 0
print_status("Classifying...")
for s in sentences:
	if (len(s) == 0): continue
	for word_index in range(len(s)):
		if is_other(s[word_index]):
			lang = 'other'
		else:
			prob_lang1 = model_lang1.get_word_log_prob(s, word_index)
			prob_lang2 = model_lang2.get_word_log_prob(s, word_index)
			if (prob_lang1 >= prob_lang2):
				lang = 'lang1'
			else:
				lang = 'lang2'
		y.append(lang)
		predictions_dict[s[word_index]] = lang

	if counter % 10000 == 0:
		print(f"{counter} of {len(sentences)}")
	counter+=1

	if (evaluation_dataset == 'test-original'):
		y.append('')

if (evaluation_dataset == 'test-original'):
	save_predictions(y, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_test_original_word_' + str(n) + '_grams.txt')
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
plt.savefig('./results/CM/confusion_matrix_' + str(n) + '_grams_words.svg',format='svg')

# Save model output
if (evaluation_dataset == 'dev'):
	save_predictions(predictions_dict, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_val_word_' + str(n) + '_grams.txt')
if (evaluation_dataset == 'test'):
	save_predictions(predictions_dict, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_test_word_' + str(n) + '_grams.txt')

# RESULTS
# Validation set
# bigrams
# 0.5092285490037218
# [0.45413354 0.31975212 0.96758294]

# trigrams
# 0.3973719523001747
# [0.39750632 0.02163962 0.96758294]

# Test set
# bigrams
# 0.5144650237257002
# [0.45770518 0.29866948 0.9704282 ]

# trigrams
# 0.4430583193020052
# [0.45181117 0.01789812 0.9704282 ]

###########################################################################
##################### RESULTS FOR WORKSHOP ################################
# Dev set
# bigrams
# Accuracy: 0.5035572321948503
# F1 score per class: [0.44502993 0.3159646  0.96758294]
# Weighted F1 score: 0.4997535295609069