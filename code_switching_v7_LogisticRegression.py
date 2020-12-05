import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.utils import is_other
from tools.utils import save_predictions
from tools.utils import printStatus
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Get training dictionaries
printStatus("Getting tokenized sentences...")
tokenized_sentences_en = pd.read_pickle(r'./dictionaries/word-level/tokenized_sentences_en.p')
tokenized_sentences_es = pd.read_pickle(r'./dictionaries/word-level/tokenized_sentences_es.p')

# Flatten lists, so we have a long array of strings (words)
tokenized_sentences_en = [item for sent in tokenized_sentences_en for item in sent][:500000]
tokenized_sentences_es = [item for sent in tokenized_sentences_es for item in sent][:500000]
X_train = tokenized_sentences_en + tokenized_sentences_es

t_train_en = ['lang1' for token in tokenized_sentences_en]
t_train_es = ['lang2' for token in tokenized_sentences_es]
t_train = t_train_en + t_train_es

# Convert a collection of text documents to a matrix of token counts
printStatus("Counting ngrams...")
count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 5))
count_data = count_vectorizer.fit_transform(X_train)

# Create and fit the SVM model
printStatus("Training Logistic Regression...")
logist_regression = LogisticRegression(max_iter=1000, random_state=123)
logist_regression.fit(count_data, t_train)

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
y = []
predictions_dict = dict()
for word in words:
	if(is_other(word)):
		lang = 'other'
	else:
		word_vect = count_vectorizer.transform([word])
		lang = logist_regression.predict(word_vect)[0]
	y.append(lang)
	predictions_dict[word] = lang

# Get accuracy
acc = accuracy_score(t, y)
print(acc)
# 0.8123908144922399 # at 1,000 words per lang
# 0.8961440109375396 # at 100,000 words per lang
# 0.9040939818214041 # at 200,000 words per lang
# 0.9079676937488923 # at 500,000 words per lang - stop here (5 min in total)

# Fq score
f1 = f1_score(t, y, average=None)
print(f1)

# Confusion matrix
conf_matrix = confusion_matrix(t, y)
classes = ['lang1', 'lang2', 'other']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
					   display_labels=classes).plot(values_format='d')
plt.savefig('./results/CM/confusion_matrix_LogisticRegression.svg', format='svg')

# Save model output
save_predictions(predictions_dict, './results/predictions/predictions_LogisticRegression.txt')