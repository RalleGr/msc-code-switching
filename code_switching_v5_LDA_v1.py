import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.utils import is_other
from tools.utils import save_predictions
from tools.utils import printStatus
from sklearn.feature_extraction.text import CountVectorizer
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


# Get data
printStatus("Getting test data...")
filepath = 'datasets/bilingual-annotated/dev.conll'
file = open(filepath, 'rt', encoding='utf8')
words = []
t = []
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
file.close()


# Remove 'other' words
printStatus("Removing 'other' data...")
words_not_other = []
for word in words:
	if(not is_other(word)):
		words_not_other.append(word)


# Convert a collection of words to a matrix of token counts
printStatus("Counting ngrams...")
count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 5))
count_data = count_vectorizer.fit_transform(words_not_other)


# Create and fit the LDA model - where the magic happens :)
printStatus("Training LDA...")
number_topics = 2
lda_model = LDA(n_components=number_topics)
lda = lda_model.fit_transform(count_data)


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
	cluster_1_label = 'lang2'
	cluster_2_label = 'lang1'


printStatus("Predicting...")
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
plt.savefig('./results/CM/confusion_matrix_LDA_v1.svg', format='svg')

# Save model output
save_predictions(predictions_dict, './results/predictions/predictions_LDA_v1.txt')


# Using 'dev' dataset
# Range 3-7 
# 0.5972605514342861
# [0.45288732 0.5623536  0.93879938]
# Range 1-3 
# 0.5677899587310429
# [0.54245257 0.39536386 0.93879938]
# Range 2-6 
# 0.5775628528749019
# [0.47643586 0.50289912 0.93879938]
# Range 2-2 
# 0.544648960680558
# [0.43280361 0.46450636 0.93879938]
# Range 3-3 
# 0.604678836367319
# [0.4516628  0.57757414 0.93879938]
# Range 2-3 
# 0.5697394738840925
# [0.47148706 0.48893527 0.93879938]
# Range 2-4 
# 0.5823986631896093
# [0.50342057 0.48884548 0.93879938]
# Range 4-4 
# 0.5630047851735575
# [0.33100593 0.56122973 0.93879938]
# Range 5-5
# 0.537888953591412
# [0.20413001 0.56551062 0.93879938]
# Range 1-4
# 0.5659670354710484
# [0.50336957 0.44520933 0.93879938]

# Using 'train' dataset
# Range 1-4
# 0.57299634321272
# [0.46323118 0.49383586 0.91109107]
# Range 3-3 - best
# 0.6626421803341617
# [0.47662609 0.66587381 0.91109107]

# ---------After adding label decision-----------
# Using 'dev' dataset
# Range 3-7 
# 0.592677924905689
# [0.46003652 0.54853954 0.93879938]
# Range 1-3 
# 0.5480669417930476
# [0.49488979 0.40493523 0.93879938]
# Range 2-6 
# 0.5523710661569233
# [0.46813166 0.44946379 0.93879938]
# Range 2-2 
# 0.5697394738840925
# [0.47148706 0.48893527 0.93879938]
# Range 3-3 
# 0.5693090614477049
# [0.4168786  0.52918593 0.93879938]
# Range 2-3
# 0.5781958123401777
# [0.48801965 0.49375368 0.93879938] 
# Range 2-4 
# 0.5650049370838292
# [0.4494762  0.49715892 0.93879938]
# Range 4-4
# 0.5631313770666126
# [0.34076382 0.55753176 0.93879938]
# Range 5-5
# 0.5361926222244727
# [0.20078417 0.56402995 0.93879938]
# Range 1-4
# 0.5605742208268982
# [0.50596277 0.42718642 0.93879938]
# Range 1-5 - best
# 0.6091855077600831
# [0.53772767 0.52085063 0.93879938]
# Range 1-6
# 0.5514849229055371
# [0.45989006 0.45556157 0.93879938]
