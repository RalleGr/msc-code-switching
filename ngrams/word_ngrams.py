from nltk.lm.preprocessing import pad_sequence
from nltk.util import everygrams
from nltk import FreqDist
import numpy as np
from numpy.lib import utils
from tools.utils import printStatus

class NGramModel:
	def __init__(self, n):
		self.n = n
		self.tokens_arr = []
		self.freq_dist = FreqDist()
	
	def train(self, tokens_arr):
		self.tokens_arr = tokens_arr
		ngrams = self.get_ngrams()
		self.freq_dist = FreqDist(ngrams)

	def get_freq(self, ngram):
		if (self.freq_dist.get(ngram) is None):
			return 1
		else:
			return self.freq_dist.get(ngram) + 1
			
	def get_ngrams(self):
		unigrams = []
		bigrams = []
		trigrams = []
		fourgrams = []
		fivegrams = []
		sixgrams = []
		printStatus("Creating n-grams...")
		j = 0
		for sent in self.tokens_arr:
			words = list(pad_sequence(
						sent,
						pad_left=True, left_pad_symbol="<s>",
						pad_right=True, right_pad_symbol="</s>",
						n=self.n))
			ngrams = list(everygrams(words, max_len=self.n))
			for ngram in ngrams:
				if (len(ngram) == 1 and self.n == 2):
					unigrams.append(ngram)
				if (len(ngram) == 2 and self.n <= 3):
					bigrams.append(ngram)
				if (len(ngram) == 3 and self.n <= 4):
					trigrams.append(ngram)
				if (len(ngram) == 4 and self.n <= 5):
					fourgrams.append(ngram)
				if (len(ngram) == 5 and self.n <= 6):
					fivegrams.append(ngram)
				if (len(ngram) == 6 and self.n <= 6):
					sixgrams.append(ngram)
			if j % (len(self.tokens_arr)/10) == 0:
				print(f"token {j} of {len(self.tokens_arr)}")
			j+=1
		return unigrams + bigrams + trigrams + fourgrams + fivegrams + sixgrams
	
	def load_ngrams_freq(self, freq_dist):
		self.freq_dist = freq_dist

	def get_word_log_prob(self, s, word_index):
		prob = 0
		if (self.n == 2):
			if (word_index == 0):
				bigram = ('<s>', s[word_index])
				unigram = ('<s>',)
			else:
				bigram = (s[word_index - 1], s[word_index])
				unigram = (s[word_index - 1],)
			prob = self.get_freq(bigram) / self.get_freq(unigram)
		return np.log(prob)