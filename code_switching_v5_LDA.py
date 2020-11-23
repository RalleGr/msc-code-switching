import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.utils import is_other
from tools.utils import printStatus
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.util import everygrams

# sources: https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
# https://github.com/kapadias/mediumposts/blob/master/nlp/published_notebooks/Introduction%20to%20Topic%20Modeling.ipynb

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
count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 4))
count_data = count_vectorizer.fit_transform(words_not_other)


# Create and fit the LDA model - where the magic happens :)
printStatus("Training LDA...")
number_topics = 2
lda_model = LDA(n_components=number_topics)
lda = lda_model.fit_transform(count_data)


words_dict = dict()
for i in range(len(words_not_other)):
	if(lda[i][0] > lda[i][1]):
		topic = 'lang1'
	else: 
		topic = 'lang2'
	words_dict[words_not_other[i]] = topic

predicted = []
for word in words:
	if(is_other(word)):
		predicted.append('other')
	else:
		predicted.append(words_dict[word])

# Get accuracy
acc = accuracy_score(t, predicted)
print(acc)

# Fq score
f1 = f1_score(t, predicted, average=None)
print(f1)

# Confusion matrix
conf_matrix = confusion_matrix(t, predicted)
classes = ['lang1', 'lang2', 'other']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
					   display_labels=classes).plot(values_format='d')
plt.savefig('./results/confusion_matrix_LDA.svg', format='svg')