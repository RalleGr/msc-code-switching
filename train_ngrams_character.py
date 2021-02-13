import pandas as pd
from models.ngrams.ngrams import NGramModel
from tools.utils import write_dict
from tools.utils import print_status
import sys
import os
from langs import langs

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"
CHAR_LEVEL_DICTIONARIES_PATH = "./dictionaries/character-level/"

# Get language code from keyboard
if len(sys.argv) == 1:
	print("Please give two letter language codes as arg, for example es en")
	print("Please give n value")
	exit(1)

# Lang 1
lang1 = sys.argv[1]
lang1_code = langs()[lang1]['code']
lang1_name = langs()[lang1]['name']

# Lang 2
lang2 = sys.argv[2]
lang2_code = langs()[lang2]['code']
lang2_name = langs()[lang2]['name']


# Get frequency dictionaries
lang1_path = WORD_LEVEL_DICTIONARIES_PATH + 'frequency_dict_' + lang1_code + '.csv'
lang2_path = WORD_LEVEL_DICTIONARIES_PATH + 'frequency_dict_' + lang2_code + '.csv'
if (os.path.exists(lang1_path) and os.path.exists(lang1_path)):
	print_status('Getting dictionaries...')
	frequency_lang1_df = pd.read_csv(lang1_path, encoding='utf-16')
	frequency_lang1_dict = frequency_lang1_df.set_index('word')['frequency'].to_dict()

	frequency_lang2_df = pd.read_csv(lang2_path, encoding='utf-16')
	frequency_lang2_dict = frequency_lang2_df.set_index('word')['frequency'].to_dict()
else:
	print("Please run: python train_probability.py " + lang1_code + " " + lang2_code)


# Create ngrams frequency dictionaries
ns = [
	2,
	3,
	4,
	5,
	6,
]
for n in ns:
	# CHARACTER NGRAMS
	print_status('Training character ngrams model... n=' + str(n))
	model_lang1 = NGramModel(n)
	model_lang1.train(frequency_lang1_dict)
	write_dict(CHAR_LEVEL_DICTIONARIES_PATH, model_lang1.freq_dist, str(n) + '_grams_dict_' + lang1_code)

	model_lang2 = NGramModel(n)
	model_lang2.train(frequency_lang2_dict)
	write_dict(CHAR_LEVEL_DICTIONARIES_PATH, model_lang2.freq_dist, str(n) + '_grams_dict_' + lang2_code)

print_status('Done!')
