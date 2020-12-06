import pandas as pd
from models.ngrams.ngrams import NGramModel
from tools.utils import write_dict
from tools.utils import printStatus

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"
CHAR_LEVEL_DICTIONARIES_PATH = "./dictionaries/character-level/"

# Get words dictionaries
frequency_en_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH+'frequency_dict_en.csv',encoding='utf-16')
frequency_en_dict = frequency_en_df.set_index('word')['frequency'].to_dict()

frequency_es_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH+'frequency_dict_es.csv',encoding='utf-16')
frequency_es_dict = frequency_es_df.set_index('word')['frequency'].to_dict()

# frequency_other_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH+'frequency_dict_other.csv',encoding='utf-16')
# frequency_other_dict = frequency_other_df.set_index('word')['frequency'].to_dict()

# Create ngrams frequency dictionaries
ns = [
	# 2,
	# 3,
	4,
	5,
	6,
]
for n in ns:
	if n!=2 and n!=3 and n!=4 and n!=5 and n!=6:
		print("n should be 2, 3, 4, 5 or 6")
		exit(1)

	# CHARACTER NGRAMS
	printStatus('Training character ngrams model...')
	model_en = NGramModel(n)
	model_en.train(frequency_en_dict)
	write_dict(CHAR_LEVEL_DICTIONARIES_PATH, model_en.freq_dist, str(n) + '_grams_dict_en')

	model_es = NGramModel(n)
	model_es.train(frequency_es_dict)
	write_dict(CHAR_LEVEL_DICTIONARIES_PATH, model_es.freq_dist, str(n) + '_grams_dict_es')

	# model_other = NGramModel(n)
	# model_other.train(frequency_other_dict)
	# write_dict(CHAR_LEVEL_DICTIONARIES_PATH, model_other.freq_dist, str(n) + '_grams_dict_other')

printStatus('Done!')
