import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.utils import save_predictions
from tools.utils import is_other

DICTIONARIES_PATH = "./dictionaries/word-level/"

# Get dictionaries
probability_en_df = pd.read_csv(DICTIONARIES_PATH+'probability_dict_en.csv', encoding='utf-16')
probability_en_dict = probability_en_df.set_index('word')['probability'].to_dict()

probability_es_df = pd.read_csv(DICTIONARIES_PATH+'probability_dict_es.csv', encoding='utf-16')
probability_es_dict = probability_es_df.set_index('word')['probability'].to_dict()

probability_other_df = pd.read_csv(DICTIONARIES_PATH+'probability_dict_other.csv', encoding='utf-8')
probability_other_dict = probability_other_df.set_index('word')['probability'].to_dict()

# Get data
# filepath = 'datasets/bilingual-annotated/dev.conll' # validation
filepath = 'datasets/bilingual-annotated/test.conll' # test
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

# Choose language with highest probability for each word
y = []
predictions_dict = dict()
for word in words:
	word = word.lower()

	# Get EN prob
	if word in probability_en_dict: prob_en = probability_en_dict[word]
	else: prob_en = probability_en_dict['OOV']

	# Get ES prob
	if word in probability_es_dict: prob_es = probability_es_dict[word]
	else: prob_es = probability_es_dict['OOV']

	# Assign class based on regex or class with highest prob
	if (is_other(word)):
		lang = 'other'
	else:
		if (prob_en >= prob_es):
			lang = 'lang1'
		else:
			lang = 'lang2'

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
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes).plot(values_format='d')
plt.savefig('./results/CM/confusion_matrix_probabilities.svg', format='svg')

# Save model output
# save_predictions(predictions_dict, './results/predictions/predictions_val_probabilities.txt')
save_predictions(predictions_dict, './results/predictions/predictions_test_probabilities.txt')

# RESULTS
# Validation (dev) set
# 0.9138921943438743
# [0.91054322 0.88774562 0.96758294]

# Test set
# 0.8930302566457472
# [0.88149196 0.85917539 0.9704282 ]