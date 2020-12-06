import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.utils import is_other
from tools.utils import save_predictions
from tools.utils import printStatus
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pandas as pd

# sources: https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
# https://github.com/kapadias/mediumposts/blob/master/nlp/published_notebooks/Introduction%20to%20Topic%20Modeling.ipynb

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"

printStatus("Getting dictionaries...")
probability_en_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH + 'probability_dict_en.csv', encoding='utf-16')
probability_en_dict = probability_en_df.set_index('word')['probability'].to_dict()

probability_es_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH + 'probability_dict_es.csv', encoding='utf-16')
probability_es_dict = probability_es_df.set_index('word')['probability'].to_dict()

# Get training dictionaries
printStatus("Getting tokenized sentences...")
tokenized_sentences_en = pd.read_pickle(r'./dictionaries/word-level/tokenized_sentences_en.p')
tokenized_sentences_es = pd.read_pickle(r'./dictionaries/word-level/tokenized_sentences_es.p')

# Flatten lists, so we have a long array of strings (words)
tokenized_sentences_en = [item for sent in tokenized_sentences_en for item in sent][:10000]
tokenized_sentences_es = [item for sent in tokenized_sentences_es for item in sent][:10000]
X_train = tokenized_sentences_en + tokenized_sentences_es

# Get 'dev' data
printStatus("Getting dev data...")
filepath = 'datasets/bilingual-annotated/dev.conll' # validation
# filepath = 'datasets/bilingual-annotated/train.conll' # test
file = open(filepath, 'rt', encoding='utf8')
dev_words = []
t = []
for line in file:
	# Remove empty lines, lines starting with # sent_enum, \n and split on tab
	if (line.strip() is not '' and '# sent_enum' not in line):
		line = line.rstrip('\n')
		splits = line.split("\t")
		if (splits[1] == 'ambiguous' or splits[1] == 'fw' or splits[1] == 'mixed' or splits[1] == 'ne' or splits[1] == 'unk'):
			continue
		else:
			dev_words.append(splits[0].lower())
			t.append(splits[1])
file.close()


# Remove 'other' words
printStatus("Removing 'other' data...")
dev_words_not_other = []
for word in dev_words:
	if(not is_other(word)):
		dev_words_not_other.append(word)


# Convert a collection of words to a matrix of token counts
printStatus("Counting ngrams...")
# tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), binary=False)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 5), binary=False)
vectorized_train_data = vectorizer.fit_transform(X_train)
vectorized_dev_data = vectorizer.transform(dev_words_not_other)


# Create and fit the LDA model
printStatus("Training LDA...")
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
	word = dev_words_not_other[i]

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
	cluster_1_label = 'lang2'
	cluster_2_label = 'lang1'

# Predict
printStatus("Predicting...")
words_dict = dict()
for i in range(len(dev_words_not_other)):
	if(lda[i][0] > lda[i][1]):
		topic = 'lang1'
	else: 
		topic = 'lang2'
	words_dict[dev_words_not_other[i]] = topic

y = []
predictions_dict = dict()
for word in dev_words:
	if(is_other(word)):
		lang = 'other'
	else:
		lang = words_dict[word]
	y.append(lang)
	predictions_dict[word] = lang

# Get accuracy
acc = accuracy_score(t, y)
print(acc)

# Fq score
f1 = f1_score(t, y, average=None)
print(f1)


# Confusion matrix
conf_matrix = confusion_matrix(t, y)
classes = ['lang1', 'lang2', 'other']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
					   display_labels=classes).plot(values_format='d')
plt.savefig('./results/CM/confusion_matrix_LDA_v2.svg', format='svg')

# Save model output
save_predictions(predictions_dict, './results/predictions/predictions_val_LDA_v2.txt')
# save_predictions(predictions_dict, './results/predictions/predictions_test_LDA_v2.txt')


# Range 3-3
# 0.5841962680709928
# [0.43054036 0.54981203 0.93879938]
# Range 1-4 - best
# 0.6342253842063954
# [0.5790509  0.54110603 0.93879938]
# Range 1-5
# 0.6334911512266754
# [0.57470909 0.54400767 0.93879938]

# ---------After adding label decision-----------
# Range 1-5 - best
# 0.6621009190571436
# [0.62412405 0.56280972 0.93879938]

# binary=True
# 0.518393802060916
# [0.44716701 0.38363071 0.93879938]

# RESULTS
# Validation set (10 000 words per language)

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
