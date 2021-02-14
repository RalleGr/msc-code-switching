import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.utils import is_other
from tools.utils import save_predictions
from tools.utils import print_status
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
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
print_status("Training SVM...")
# svm = LinearSVC()
svm = SVC(random_state=123)
svm.fit(vectorized_data, t_train)

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

# Create the array of y classes
print_status("Predicting...")
y = []
predictions_dict = dict()
for word in words:
	if (word != ''):
		if(is_other(word)):
			lang = 'other'
		else:
			word_vect = vectorizer.transform([word])
			lang = svm.predict(word_vect)[0]
		y.append(lang)
		predictions_dict[word] = lang
	else:
		y.append('')

if (evaluation_dataset == 'test-original'):
	save_predictions(y, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_test_original_SVM.txt')
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
plt.savefig('./results/CM/confusion_matrix_SVM.svg', format='svg')

# Save model output
if (evaluation_dataset == 'dev'):
	save_predictions(predictions_dict, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_val_SVM.txt')
if (evaluation_dataset == 'test'):
	save_predictions(predictions_dict, './results/predictions/' + lang1_code + '-' + lang2_code + '/predictions_test_SVM.txt')


# RESULTS
# Validation set

# 	CountVectorizer(binary=False) - 10 000 words per lang
# 0.8524951262121174
# [0.83348071 0.8130638  0.96758294]

# 	CountVectorizer(binary=True) - 10 000 words per lang
# 0.8487986429349065
# [0.82817932 0.80917739 0.96758294]

# 	TfidfVectorizer(binary=False) - 10 000 words per lang
# 0.8956629617439299
# [0.88561258 0.86852914 0.96758294]
# 	TfidfVectorizer(binary=True) - 10 000 words per lang
# 0.8939919487556017
# [0.88348477 0.86650164 0.96758294]

# 	TfidfVectorizer(binary=True) - 100 000 words per lang (saved as .txt)
# 0.9142466516444286
# [0.90836723 0.89206897 0.96758294]

# Test set

# TfidfVectorizer(binary=True) - 100 000 words per lang
# 0.9044594111944487
# [0.89132306 0.88025611 0.9704282 ]

###########################################################################
##################### RESULTS FOR WORKSHOP ################################
# Dev set
# TfidfVectorizer(binary=True) - 100 000 words per lang
# Accuracy: 0.9142466516444286
# F1 score per class: [0.90836723 0.89206897 0.96758294]
# Weighted F1 score: 0.913935210077486