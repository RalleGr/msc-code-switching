from nltk.lm.preprocessing import pad_both_ends
from nltk.util import everygrams
from nltk import FreqDist
from collections import Counter
from nltk.lm.preprocessing import flatten
import numpy as np

class NGramModel:
	def __init__(self):
		self.tokens_dict = dict()
		self.freq_dist = FreqDist()
	
	def train(self, tokens_dict):
		self.tokens_dict = tokens_dict
		ngrams = self.get_ngrams()
		self.freq_dist = FreqDist(ngrams)

	def get_freq(self, ngram):
		if (self.freq_dist.get(ngram) is None):
			return 1
		else:
			return self.freq_dist.get(ngram) + 1
			
	def get_ngrams(self, largest_ngram = 2):
		unigrams = []
		bigrams = []
		for token in self.tokens_dict.keys():
			if type(token) is float:
				print(f"ERROR : unknown token {token}")
				continue
			chars = list(pad_both_ends(str(token), n=largest_ngram))
			ngrams = list(everygrams(chars, max_len=largest_ngram))
			for ngram in ngrams:
				if (len(ngram) == 1):
					for i in range(self.tokens_dict[token]):
						unigrams.append(ngram)
				if (len(ngram) == 2):
					for i in range(self.tokens_dict[token]):
						bigrams.append(ngram)
		return unigrams + bigrams
	
	def load_ngrams_freq(self, ngrams):
		self.freq_dist = FreqDist(ngrams)

	def get_word_log_prob(self, word):
		word_log_prob = 0
		for i in range(len(word)):
			if (i == 0):
				bigram = ('<s>', word[i])
				prob = self.get_freq(bigram) / self.get_freq(('<s>',))
			elif (i == len(word) - 1):
				bigram = (word[i], ('</s>',))
				c = word[i]
				prob = self.get_freq(bigram) / self.get_freq((c,))
			else:
				bigram = (word[i], word[i + 1])
				c = word[i]
				prob = self.get_freq(bigram) / self.get_freq((c,))
			# print(prob)
			word_log_prob += np.log(prob)
		return word_log_prob