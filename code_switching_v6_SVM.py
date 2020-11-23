import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.utils import is_other
from tools.utils import printStatus
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pandas as pd

# Get training dictionaries
printStatus("Getting tokenized sentences...")
tokenized_sentences_en = pd.read_pickle(r'tokenized_sentences_en.p')
tokenized_sentences_es = pd.read_pickle(r'tokenized_sentences_es.p')

# Flatten lists, so we have a long array of strings (words)
tokenized_sentences_en = [item for sent in tokenized_sentences_en for item in sent][:100000]
tokenized_sentences_es = [item for sent in tokenized_sentences_es for item in sent][:100000]
X_train = tokenized_sentences_en + tokenized_sentences_es

t_train_en = ['lang1' for token in tokenized_sentences_en]
t_train_es = ['lang2' for token in tokenized_sentences_es]
t_train = t_train_en + t_train_es

# Convert a collection of text documents to a matrix of token counts
printStatus("Counting ngrams...")
count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 5))
count_data = count_vectorizer.fit_transform(X_train)

# Create and fit the SVM model
printStatus("Training SVM...")
# svm = LinearSVC()
svm = SVC()
svm.fit(count_data, t_train)

# Get test data
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

# Create the array of predicted classes
printStatus("Predicting...")
predicted = []
for word in words:
	if(is_other(word)):
		predicted.append('other')
	else:
		word_vect = count_vectorizer.transform([word])
		y = svm.predict(word_vect)[0]
		predicted.append(y)

# Get accuracy
acc = accuracy_score(t, predicted)
print(acc)
# 0.8240580297237765 # at 1,000 words per lang
# 0.8883965870825632 # at 100,000 words per lang - stop here (55 min in total)

# Fq score
f1 = f1_score(t, predicted, average=None)
print(f1)

# Confusion matrix
conf_matrix = confusion_matrix(t, predicted)
classes = ['lang1', 'lang2', 'other']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
					   display_labels=classes).plot(values_format='d')
plt.savefig('./results/confusion_matrix_SVM.svg', format='svg')