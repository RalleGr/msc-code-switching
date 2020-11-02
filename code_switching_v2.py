import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from ngrams.ngrams import NGramModel
from tools.utils import is_other

CHAR_LEVEL_DICTIONARIES_PATH = "./dictionaries/character-level/"

# Get dictionaries
print("Getting dictionaries...")
frequency_en_df = pd.read_csv(CHAR_LEVEL_DICTIONARIES_PATH+'frequency_dict_en.csv',encoding='utf-16', converters={"word": ast.literal_eval})
frequency_en_dict = frequency_en_df.set_index('word')['frequency'].to_dict()

frequency_es_df = pd.read_csv(CHAR_LEVEL_DICTIONARIES_PATH+'frequency_dict_es.csv',encoding='utf-16', converters={"word": ast.literal_eval})
frequency_es_dict = frequency_es_df.set_index('word')['frequency'].to_dict()

# Apply ngram model to en, es and other
print("Applying NGRAM model...")
model_en = NGramModel()
model_en.load_ngrams_freq(frequency_en_dict)

model_es = NGramModel()
model_es.load_ngrams_freq(frequency_es_dict)

# Get data
print("Getting test data...")
filepath = 'datasets/bilingual-annotated/dev.conll'
file = open(filepath, 'rt', encoding='utf8')
words = []
t = []
for line in file:
	# Remove empty lines, lines starting with # sent_enum, \n and split on tab
	if (line.strip() is not '' and '# sent_enum' not in line):
		line = line.rstrip('\n')
		splits = line.split("\t")
		words.append(splits[0])
		t.append(splits[1])
file.close()

# Choose language with highest probability for each word based on ngrams
y = []
counter = 0
print("Classifying...")
for word in words:
	word = word.lower()
	if is_other(word):
		lang = 'other'
	else:
		prob_en = model_en.get_word_log_prob(word)
		prob_es = model_es.get_word_log_prob(word)
		if (prob_en >= prob_es):
			lang = 'lang1'
		else:
			lang = 'lang2'
	
	y.append(lang)

	if counter % 10000 == 0:
		print(f"{counter} of {len(words)}")
	counter+=1

# Get accuracy
acc = accuracy_score(t, y)
print(acc)
# 0.3306321355653757 # first try with unigrams and bigrams
# 0.7512812260156966 # after fixing bug

# Fq score
f1 = f1_score(t, y, average=None)
print(f1)

# Confusion matrix
conf_matrix = confusion_matrix(t, y)
classes = ['ambiguous', 'fw', 'lang1', 'lang2', 'mixed', 'ne', 'other', 'unk']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes).plot(values_format='d')
plt.savefig("confusion_matrix_ngrams.png")