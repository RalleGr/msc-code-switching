from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from models.viterbi.viterbi_identifier import ViterbiIdentifier
from tools.utils import is_other
from tools.utils import print_status
from tools.utils import save_predictions
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Based on eginhard’s implementation (Source: https://github.com/eginhard/word-level-language-id)
# This version was used as final model

WORD_DICTIONARIES_PATH = "./dictionaries/word-level/"
VITERBI_DICTIONARIES_PATH = "./dictionaries/viterbi/"

# Language codes
en = 'en'
es = 'es'

# Language model files.
ngram = 2
en_lm = VITERBI_DICTIONARIES_PATH + str(ngram) + '-gram-en.lm'
es_lm = VITERBI_DICTIONARIES_PATH + str(ngram) + '-gram-es.lm'

# Unigram frequency lexicons.
en_lex = WORD_DICTIONARIES_PATH + 'frequency_dict_en.csv'
es_lex = WORD_DICTIONARIES_PATH + 'frequency_dict_es.csv'

identifier = ViterbiIdentifier(en, es,
								en_lm, es_lm,
								en_lex, es_lex)

# Get data
print_status("Getting test data...")
# filepath = './datasets/bilingual-annotated/dev.conll' # validation
filepath = './datasets/bilingual-annotated/test.conll' # test
# filepath = './datasets/bilingual-annotated/test-original.conll' # original test set from LinCE
file = open(filepath, 'rt', encoding='utf8')
sentences = []
t = []
s = []
# Own test set
for line in file:
	# Remove empty lines, lines starting with # sent_enum, \n and split on tab
	if (line.strip() is not ''):
		if ('# sent_enum' in line):
			s = []
		else:
			line = line.rstrip('\n')
			splits = line.split("\t")
			if (splits[1]=='ambiguous' or splits[1]=='fw' or splits[1]=='mixed' or splits[1]=='ne' or splits[1]=='unk'):
				continue
			else:
				s.append(splits[0].lower())
				t.append(splits[1])
	else:
		sentences.append(s)

"""
# Original test set
for line in file:
	# Remove empty lines, lines starting with # sent_enum, \n and split on tab
	if (line.strip() is not ''):
		token = line.rstrip('\n')
		s.append(token.lower())
	else:
		sentences.append(s)
		s = []
"""

file.close()

y = []
predictions_dict = dict()
for tokens in sentences:
	if(len(tokens) > 0):
		# Separate 'lang' words from 'other' words
		lang_tokens = []
		other_indexes = []
		for i in range(len(tokens)):
			if (is_other(tokens[i])): other_indexes.append(i)
			else: lang_tokens.append(tokens[i])
		
		# For sentences with 'lang1', 'lang2' and 'other' words
		if(len(lang_tokens) > 0):
			y_sentence = identifier.identify(lang_tokens)
			for index in other_indexes:
				y_sentence.insert(index, 'other')

		# For sentences that are made up only of 'other' words
		else:
			y_sentence = []
			for index in other_indexes:
				y_sentence.append('other')
		for i in range(len(tokens)):
			predictions_dict[tokens[i]] = y_sentence[i]
		y.append(y_sentence)

save_predictions(y, './results/predictions/predictions_test_original_viterbi_v1.txt')

# Own test set with labels
# Flatten y list
y = [item for y_sent in y for item in y_sent]

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
plt.savefig('./results/CM/confusion_matrix_' + 'viterbi_v1.svg', format='svg')

# Save model output
# save_predictions(predictions_dict, './results/predictions/predictions_val_viterbi_v1.txt')
save_predictions(predictions_dict, './results/predictions/predictions_test_viterbi_v1.txt')

# RESULTS
# Validation set
# 0.9576423525837405
# [0.95817377 0.9517566  0.96758294]

# Test set
# 0.9368080004081841
# [0.92870999 0.92623526 0.9704282 ]
