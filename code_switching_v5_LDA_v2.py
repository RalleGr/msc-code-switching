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

# Get training dictionaries
printStatus("Getting tokenized sentences...")
tokenized_sentences_en = pd.read_pickle(r'./dictionaries/word-level/tokenized_sentences_en.p')
tokenized_sentences_es = pd.read_pickle(r'./dictionaries/word-level/tokenized_sentences_es.p')

# Flatten lists, so we have a long array of strings (words)
tokenized_sentences_en = [item for sent in tokenized_sentences_en for item in sent][:100000]
tokenized_sentences_es = [item for sent in tokenized_sentences_es for item in sent][:100000]
X_train = tokenized_sentences_en + tokenized_sentences_es

# Get 'dev' data
printStatus("Getting dev data...")
filepath = 'datasets/bilingual-annotated/dev.conll'
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
count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 4))
count_train_data = count_vectorizer.fit_transform(X_train)
count_dev_data = count_vectorizer.transform(dev_words_not_other)


# Create and fit the LDA model - where the magic happens :)
printStatus("Training LDA...")
number_topics = 2
lda_model = LDA(n_components=number_topics)
lda_model.fit(count_train_data)
lda = lda_model.transform(count_dev_data)

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
save_predictions(predictions_dict, './results/predictions/predictions_LDA_v2.txt')

# Range 3-3
# 0.5841962680709928
# [0.43054036 0.54981203 0.93879938]
# Range 1-4 - best
# 0.6342253842063954
# [0.5790509  0.54110603 0.93879938]
# Range 1-5
# 0.6334911512266754
# [0.57470909 0.54400767 0.93879938]
