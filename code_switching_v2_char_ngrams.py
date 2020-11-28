import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from models.ngrams.ngrams import NGramModel
from tools.utils import save_predictions
from tools.utils import is_other
from tools.utils import printStatus
import sys

CHAR_LEVEL_DICTIONARIES_PATH = "./dictionaries/character-level/"

n = int(sys.argv[1])
if n!=2 and n!=3 and n!=4 and n!=5 and n!=6:
	print("n should be 2, 3, 4, 5 or 6")
	exit(1)

# Get dictionaries
printStatus("Getting dictionaries...")
frequency_en_df = pd.read_csv(CHAR_LEVEL_DICTIONARIES_PATH + str(n) + '_grams_dict_en.csv', encoding='utf-16', converters={"word": ast.literal_eval})
frequency_en_dict = frequency_en_df.set_index('word')['frequency'].to_dict()

frequency_es_df = pd.read_csv(CHAR_LEVEL_DICTIONARIES_PATH + str(n) + '_grams_dict_es.csv', encoding='utf-16', converters={"word": ast.literal_eval})
frequency_es_dict = frequency_es_df.set_index('word')['frequency'].to_dict()

# Apply ngram model to en, es and other
printStatus("Applying NGRAM model...")
model_en = NGramModel(n)
model_en.load_ngrams_freq(frequency_en_dict)

model_es = NGramModel(n)
model_es.load_ngrams_freq(frequency_es_dict)

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
		if (splits[1]=='ambiguous' or splits[1]=='fw' or splits[1]=='mixed' or splits[1]=='ne' or splits[1]=='unk'):
			continue
		else:
			words.append(splits[0])
			t.append(splits[1])
file.close()

# Choose language with highest probability for each word based on ngrams
y = []
predictions_dict = dict()
counter = 0
printStatus("Classifying...")
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
	predictions_dict[word] = lang

	if counter % 10000 == 0:
		print(f"{counter} of {len(words)}")
	counter+=1

# Get accuracy
acc = accuracy_score(t, y)
print(acc)
# 0.3306321355653757 # first try with unigrams and bigrams
# 0.7512812260156966 # after fixing bug (string type vs. tuple type)
# 0.7955034559586804 # using unigrams and bigrams
# 0.8777628680659291 # using bigrams and trigrams
# 0.9003721801655822 # using trigrams grams and 4 grams
# 0.9113856748613819 # using 4 grams and 5 grams - best
# 0.9095374332227764 # using 5 grams and 6 grams

# Fq score
f1 = f1_score(t, y, average=None)
print(f1)

# Confusion matrix
conf_matrix = confusion_matrix(t, y)
classes = ['lang1', 'lang2', 'other']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes).plot(values_format='d')
plt.savefig('./results/CM/confusion_matrix_char_' + str(n) + '_grams.svg', format='svg')

# Save model output
save_predictions(predictions_dict, './results/predictions/predictions_char_' + str(n) + '_grams.txt')