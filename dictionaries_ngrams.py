import pandas as pd
import os
from ngrams.ngrams import NGramModel
from tools.utils import write_dict

WORD_LEVEL_DICTIONARIES_PATH = "./dictionaries/word-level/"
CHAR_LEVEL_DICTIONARIES_PATH = "./dictionaries/character-level/"

# Get words dictionaries
frequency_en_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH+'frequency_dict_en.csv',encoding='utf-16')
frequency_en_dict = frequency_en_df.set_index('word')['frequency'].to_dict()

frequency_es_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH+'frequency_dict_es.csv',encoding='utf-16')
frequency_es_dict = frequency_es_df.set_index('word')['frequency'].to_dict()

frequency_other_df = pd.read_csv(WORD_LEVEL_DICTIONARIES_PATH+'frequency_dict_other.csv',encoding='utf-16')
frequency_other_dict = frequency_other_df.set_index('word')['frequency'].to_dict()

# Create ngrams frequency dictionaries
model_en = NGramModel().train(frequency_en_dict)
write_dict(CHAR_LEVEL_DICTIONARIES_PATH, model_en.freq_dist, 'en')

model_es = NGramModel().train(frequency_es_dict)
write_dict(CHAR_LEVEL_DICTIONARIES_PATH, model_es.freq_dist, 'es')

model_other = NGramModel().train(frequency_other_dict)
write_dict(CHAR_LEVEL_DICTIONARIES_PATH, model_other.freq_dist, 'other')