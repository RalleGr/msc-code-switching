import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.utils import is_other
from tools.utils import save_predictions
from tools.utils import print_status
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import sys
import os

# Get language codes and evaluation dataset from keyboard
if len(sys.argv) == 1:
	print("Please give two letter language codes as arg, for example en es")
	print("Please enter evaluation dataset: 'dev', 'test' or 'test-original'")
	exit(1)
lang1_code = sys.argv[1]
lang2_code = sys.argv[2]
evaluation_dataset = sys.argv[3]

# Get training dictionaries
print_status("Getting tokenized sentences...")
lang1_path_tokenized = './dictionaries/word-level/tokenized_sentences_' + lang1_code + '.p'
lang2_path_tokenized = './dictionaries/word-level/tokenized_sentences_' + lang2_code + '.p'

if (os.path.exists(lang1_path_tokenized) and os.path.exists(lang2_path_tokenized)):
	tokenized_sentences_lang1 = pd.read_pickle(lang1_path_tokenized)
	tokenized_sentences_lang2 = pd.read_pickle(lang2_path_tokenized)
else:
	print("Please run: python train_ngrams_word.py " + lang1_code + " " + lang2_code + " 2")

# Flatten lists, so we have a long array of strings (words)
tokenized_sentences_lang1 = [item for sent in tokenized_sentences_lang1 for item in sent][:100000]
tokenized_sentences_lang2 = [item for sent in tokenized_sentences_lang2 for item in sent][:100000]
X_train = tokenized_sentences_lang1 + tokenized_sentences_lang2

t_train_lang1 = ['lang1' for token in tokenized_sentences_lang1]
t_train_lang2 = ['lang2' for token in tokenized_sentences_lang2]
t_train = t_train_lang1 + t_train_lang2

# Convert a collection of text documents to a matrix of token counts
print_status("Counting ngrams...")
# vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 5), binary=True)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), binary=True)
vectorized_data = vectorizer.fit_transform(X_train)

# Create and fit the SVM model
print_status("Training Logistic Regression...")
logist_regression = LogisticRegression(max_iter=1000, random_state=123)
logist_regression.fit(vectorized_data, t_train)

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
			if (splits[1] == 'ambiguous' or splits[1] == 'fw' or splits[1] == 'mixed' or splits[1] == 'ne' or splits[1] == 'unk'):
				continue
			else:
				words.append(splits[0].lower())
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

# Create the array of predicted classes
print_status("Predicting...")
y = []
predictions_dict = dict()
for word in words:
	if (word != ''):
		if(is_other(word)):
			lang = 'other'
		else:
			word_vect = vectorizer.transform([word])
			lang = logist_regression.predict(word_vect)[0]
		y.append(lang)
		predictions_dict[word] = lang
	else:
		y.append('')

if (evaluation_dataset == 'test-original'):
	save_predictions(y, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_test_original_LogisticRegression.txt')
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
ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
					   display_labels=classes).plot(values_format='d')
plt.savefig('./results/CM/confusion_matrix_LogisticRegression.svg', format='svg')

# Save model output
if (evaluation_dataset == 'dev'):
	save_predictions(predictions_dict, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_val_LogisticRegression.txt')
if (evaluation_dataset == 'test'):
	save_predictions(predictions_dict, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_test_LogisticRegression.txt')

# RESULTS
# Validation set
# 0.8123908144922399 # at 1,000 words per lang
# 0.8961440109375396 # at 100,000 words per lang
# 0.9040939818214041 # at 200,000 words per lang
# 0.9079676937488923 # at 500,000 words per lang - stop here (5 min in total)

# 	CountVectorizer(binary=False)
# 0.9200951971035775
# [0.91724097 0.89768954 0.96758294]
# 	CountVectorizer(binary=True)
# 0.920474972782743
# [0.9174561  0.89851113 0.96758294]

# 	TfidfVectorizer(binary=False)
# 0.9235638149732891
# [0.92139776 0.90215224 0.96758294]
# 	TfidfVectorizer(binary=True)
# 0.9241461376813429
# [0.9219929  0.90305071 0.96758294]

# Test set
# CountVectorizer(binary=True)
# 0.899943874687484
# [0.89029626 0.86940129 0.9704282 ]

# TfidfVectorizer(binary=True)
# 0.9110924026736058
# [0.90466873 0.88314996 0.9704282 ]

###########################################################################
##################### RESULTS FOR WORKSHOP ################################
# Dev set
# TfidfVectorizer(binary=True) - 100 000 words per lang
# Accuracy: 0.9037395245208497
# F1 score per class: [0.89650742 0.87737252 0.96758294]
# Weighted F1 score: 0.9033524655078986