import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.utils import is_other
from tools.utils import save_predictions
from tools.utils import print_status
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pandas as pd
import sys

# sources: https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
# https://github.com/kapadias/mediumposts/blob/master/nlp/published_notebooks/Introduction%20to%20Topic%20Modeling.ipynb

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"

# Get evaluation dataset from keyboard
if len(sys.argv) == 1:
	print("Please enter evaluation dataset: 'dev', 'test' or 'test-original'")
	exit(1)
evaluation_dataset = sys.argv[1]

print_status("Getting dictionaries...")
probability_en_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH + 'probability_dict_en.csv', encoding='utf-16')
probability_en_dict = probability_en_df.set_index('word')['probability'].to_dict()

probability_es_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH + 'probability_dict_es.csv', encoding='utf-16')
probability_es_dict = probability_es_df.set_index('word')['probability'].to_dict()

# Get training dictionaries
print_status("Getting tokenized sentences...")
tokenized_sentences_en = pd.read_pickle(r'./dictionaries/word-level/tokenized_sentences_en.p')
tokenized_sentences_es = pd.read_pickle(r'./dictionaries/word-level/tokenized_sentences_es.p')

# Flatten lists, so we have a long array of strings (words)
tokenized_sentences_en = [item for sent in tokenized_sentences_en for item in sent][:100000]
tokenized_sentences_es = [item for sent in tokenized_sentences_es for item in sent][:100000]
X_train = tokenized_sentences_en + tokenized_sentences_es

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


# Remove 'other' words
print_status("Removing 'other' data...")
words_not_other = []
for word in words:
	if(word != '' and not is_other(word)):
		words_not_other.append(word)


# Convert a collection of words to a matrix of token counts
print_status("Counting ngrams...")
# vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 5), binary=True)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), binary=True)
vectorized_train_data = vectorizer.fit_transform(X_train)
vectorized_dev_data = vectorizer.transform(words_not_other)


# Create and fit the LDA model
print_status("Training LDA...")
number_topics = 2
lda_model = LDA(n_components=number_topics, max_iter=100, random_state=123)
lda_model.fit(vectorized_train_data)
lda = lda_model.transform(vectorized_dev_data)

# Decide labels that belong to each cluster
cluster_0_label = ''
cluster_1_label = ''
# Get indexes of words that represent better cluster 0
cluster_0 = lda[:,0]
top_n_words_c0_idx = (-cluster_0).argsort()[:10]
# Check in which language these words belong to
count_lang1 = 0
count_lang2 = 0

for i in top_n_words_c0_idx:
	word = words_not_other[i]

	# Get EN prob
	if word in probability_en_dict: prob_en = probability_en_dict[word]
	else: prob_en = probability_en_dict['OOV']

	# Get ES prob
	if word in probability_es_dict: prob_es = probability_es_dict[word]
	else: prob_es = probability_es_dict['OOV']

	# Assign class based on regex or class with highest prob
	if (prob_en >= prob_es):
		count_lang1 += 1
	else:
		count_lang2 += 1

if(count_lang1>=count_lang2):
	cluster_0_label = 'lang1'
	cluster_1_label = 'lang2'
else: 
	cluster_0_label = 'lang2'
	cluster_1_label = 'lang1'

# Predict
print_status("Predicting...")
words_dict = dict()
for i in range(len(words_not_other)):
	if(lda[i][0] > lda[i][1]):
		topic = cluster_0_label
	else:
		topic = cluster_1_label
	words_dict[words_not_other[i]] = topic

y = []
predictions_dict = dict()
for word in words:
	if (word != ''):
		if(is_other(word)):
			lang = 'other'
		else:
			lang = words_dict[word]
		y.append(lang)
		predictions_dict[word] = lang
	else:
		y.append('')

if (evaluation_dataset == 'test-original'):
	save_predictions(y, './results/predictions/predictions_test_original_LDA.txt')
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
plt.savefig('./results/CM/confusion_matrix_LDA.svg', format='svg')

# Save model output
if (evaluation_dataset == 'dev'):
	save_predictions(predictions_dict, './results/predictions/predictions_val_LDA.txt')
if (evaluation_dataset == 'test'):
	save_predictions(predictions_dict, './results/predictions/predictions_test_LDA.txt')

# RESULTS

# Validation set

# 	CountVectorizer(binary=False) - 10 000 words per lang
# 0.5350532951869762
# [0.46861357 0.37814753 0.96758294]

# 	CountVectorizer(binary=True)
# 0.583613945362939 - 10 000 words per lang
# [0.54220152 0.4184283  0.96758294]

# 	CountVectorizer(binary=False)
# 0.6311365420158493 - 20 000 words per lang
# [0.58501096 0.50172747 0.96758294]

# 	CountVectorizer(binary=False)
# 0.5337114211205914 - 30 000 words per lang
# [0.41353994 0.43592911 0.96758294]

# 	CountVectorizer(binary=False)
# 0.5170519279945313 ---- with 100 000 words per language. why does it give worse results?
# [0.39075903 0.41694193 0.96758294]

# 	TfidfVectorizer(binary=False) - 20 000 words per lang
# 0.637694002076107
# [0.58269743 0.52399839 0.96758294]
# 	TfidfVectorizer(binary=True) - 20 000 words per lang
# 0.6386814188419374
# [0.58867724 0.51898777 0.96758294]

# Test set

# TfidfVectorizer(binary=False)
# 0.6481198020307158
# [0.5947801  0.51802328 0.9704282 ]

###########################################################################
##################### RESULTS FOR WORKSHOP ################################
# Dev set
# TfidfVectorizer(binary=True) - 100 000 words per lang
# 0.6453148340380283
# [0.59368086 0.53088275 0.96758294]