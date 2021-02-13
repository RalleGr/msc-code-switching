from models.viterbi.viterbi_language_model import ViterbiLanguageModel
import sys
import os
from langs import langs
from tools.utils import print_status

# Train Viterbi version 1

WORD_DICTIONARIES_PATH = "./dictionaries/word-level/"
VITERBI_DICTIONARIES_PATH = "./dictionaries/viterbi/"

# Get language code from keyboard
if len(sys.argv) == 1:
	print("Please give two letter language codes as arg, for example en es")
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
lang1_path = WORD_DICTIONARIES_PATH + 'frequency_dict_' + lang1_code + '.csv'
lang2_path = WORD_DICTIONARIES_PATH + 'frequency_dict_' + lang2_code + '.csv'
if (not os.path.exists(lang1_path) or not os.path.exists(lang1_path)):
	print("Please run: python train_probability.py " + lang1_code + " " + lang2_code)

# n value for ngrams
ngram = 2

# Create models
lang1_lm = ViterbiLanguageModel(lang1_code, ngram, lang1_path)
lang2 = ViterbiLanguageModel(lang2_code, ngram, lang2_path)

# Train and save character n-gram models.
lang1_lm.train()
lang1_lm_fn = VITERBI_DICTIONARIES_PATH + str(ngram) + '-gram-' + lang1_code + '.lm'
lang1_lm.dump(lang1_lm_fn)

lang2.train()
lang2_fn = VITERBI_DICTIONARIES_PATH + str(ngram) + '-gram-' + lang2_code + '.lm'
lang2.dump(lang2_fn)
