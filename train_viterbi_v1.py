from models.viterbi.viterbi_language_model import ViterbiLanguageModel

# Train Viterbi version 1

WORD_DICTIONARIES_PATH = "./dictionaries/word-level/"
VITERBI_DICTIONARIES_PATH = "./dictionaries/viterbi/"

# Language codes
en = 'lang1'
es = 'lang2'

# Unigram frequency lexicons.
en_lex = WORD_DICTIONARIES_PATH + 'frequency_dict_en.csv'
es_lex = WORD_DICTIONARIES_PATH + 'frequency_dict_es.csv'

# n value for ngrams
ngram = 2

# Create models
en_lm = ViterbiLanguageModel(en, ngram, en_lex)
es_lm = ViterbiLanguageModel(es, ngram, es_lex)

# Train and save character n-gram models.
en_lm.train()
en_lm_fn = VITERBI_DICTIONARIES_PATH + str(ngram) + '-gram-en.lm'
en_lm.dump(en_lm_fn)

es_lm.train()
es_lm_fn = VITERBI_DICTIONARIES_PATH + str(ngram) + '-gram-es.lm'
es_lm.dump(es_lm_fn)
