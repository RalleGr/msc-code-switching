import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from ngrams.ngrams import NGramModel
from tools.utils import is_other
import sys

CHAR_LEVEL_DICTIONARIES_PATH = "./dictionaries/character-level/"

n = int(sys.argv[1])
if n!=2 and n!=3 and n!=4 and n!=5:
	print("n should be 2, 3, 4 or 5")
	exit(1)

# Get dictionaries
print("Getting dictionaries...")
frequency_en_df = pd.read_csv(CHAR_LEVEL_DICTIONARIES_PATH + str(n) + '_grams_dict_en.csv', encoding='utf-16', converters={"word": ast.literal_eval})
frequency_en_dict = frequency_en_df.set_index('word')['frequency'].to_dict()

frequency_es_df = pd.read_csv(CHAR_LEVEL_DICTIONARIES_PATH + str(n) + '_grams_dict_es.csv', encoding='utf-16', converters={"word": ast.literal_eval})
frequency_es_dict = frequency_es_df.set_index('word')['frequency'].to_dict()

# Apply ngram model to en, es and other
print("Applying NGRAM model...")
model_en = NGramModel(n)
model_en.load_ngrams_freq(frequency_en_dict)

model_es = NGramModel(n)
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
		if (splits[1]=='ambiguous' or splits[1]=='fw' or splits[1]=='mixed' or splits[1]=='ne'):
			t.append('unk')
		else:
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
		# print(f"EN: {word} : {prob_en}")
		# print(f"ES: {word} : {prob_es}")
		if(prob_en < -30 and prob_es < -30):
			lang = 'unk'
		elif (prob_en >= prob_es):
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
# 0.7512812260156966 # after fixing bug (string type vs. tuple type)
# 0.7647990889059444 # using bigrams and trigrams
# 0.7778960659552870 # using unigrams and bigrams, fixed bug when ngrams are skipped (explain on old code)
# 0.8583347775494541 # using bigrams and trigrams on fixed code
# 0.8786610878661087 # using trigrams and 4 grams
# 0.8901983115050383 # using 4 grams and 5 grams

# Fq score
f1 = f1_score(t, y, average=None)
print(f1)

# Confusion matrix
conf_matrix = confusion_matrix(t, y)
classes = ['lang1', 'lang2', 'other', 'unk']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes).plot(values_format='d')
plt.savefig('confusion_matrix_' + str(n) + '_grams.png')