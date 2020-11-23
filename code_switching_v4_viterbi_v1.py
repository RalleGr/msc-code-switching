from os import closerange
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from models.viterbi.viterbi_identifier import ViterbiIdentifier
from tools.utils import is_other
from tools.utils import printStatus

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
printStatus("Getting test data...")
filepath = './datasets/bilingual-annotated/dev.conll'
file = open(filepath, 'rt', encoding='utf8')
sentences = []
t = []
s = []
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
file.close()

y = []
for tokens in sentences:
	if(len(tokens) > 0):
		lang_tokens = []
		other_indexes = []
		for i in range(len(tokens)):
			if (is_other(tokens[i])): other_indexes.append(i)
			else: lang_tokens.append(tokens[i])
		if(len(lang_tokens) > 0):
			y_sentence = identifier.identify(lang_tokens)
			for index in other_indexes:
				y_sentence.insert(index, 'other')
		else:
			y_sentence = []
			for index in other_indexes:
				y_sentence.append('other')
		y.append(y_sentence)

# Flatten y list
y = [item for y_sent in y for item in y_sent]

# Get accuracy
acc = accuracy_score(t, y)
print(acc)
# 0.9447299794921133 # with 2grams
# 0.9437678811048941 # with 3grams
# 0.9401220345849052 # with 4grams - stop here

# Fq score
f1 = f1_score(t, y, average=None)
print(f1)