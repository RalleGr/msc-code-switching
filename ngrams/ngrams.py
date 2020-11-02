from nltk.lm.preprocessing import pad_sequence
from nltk.util import everygrams
from nltk import FreqDist
import numpy as np

class NGramModel:
	def __init__(self, n):
		self.n = n
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
			
	def get_ngrams(self):
		unigrams = []
		bigrams = []
		trigrams = []
		for token in self.tokens_dict.keys():
			if type(token) is float:
				print(f"ERROR : unknown token {token}")
				continue
			chars = list(pad_sequence(
						str(token),
						pad_left=True, left_pad_symbol="<w>",
						pad_right=True, right_pad_symbol="</w>",
						n=self.n))
			ngrams = list(everygrams(chars, max_len=self.n))
			for ngram in ngrams:
				if (len(ngram) == 1 and self.n == 2):
					for i in range(self.tokens_dict[token]):
						unigrams.append(ngram)
				if (len(ngram) == 2 and self.n <= 3):
					for i in range(self.tokens_dict[token]):
						bigrams.append(ngram)
				if (len(ngram) == 3 and self.n <= 3):
					for i in range(self.tokens_dict[token]):
						trigrams.append(ngram)
		return unigrams + bigrams + trigrams
	
	def load_ngrams_freq(self, freq_dist):
		self.freq_dist = freq_dist

	def get_word_log_prob(self, word):
		word_log_prob = 0
		for i in range(len(word)):
			prob = 0
			if (self.n == 2):
				if (i == 0):
					bigram = ('<w>', word[i])
					unigram = ('<w>',)
				elif (i == len(word) - 1):
					bigram = (word[i], ('</w>',))
					unigram = (word[i],)
				else:
					bigram = (word[i], word[i+1])
					unigram = (word[i],)
				prob = self.get_freq(bigram) / self.get_freq(unigram)
			
			elif (self.n == 3):
				if (i == 0):
					trigram = ('<w>', '<w>', word[i])
					bigram = ('<w>', '<w>')
				elif (i == 1):
					trigram = ('<w>', word[i], word[i+1]) if len(word) > 2 else ('<w>', word[i], '</w>')
					bigram = ('<w>', word[i])
				elif (i == len(word) - 2):
					trigram = (word[i], word[i+1], '</w>')
					bigram = (word[i], word[i+1])
				elif (i == len(word) -1):
					trigram = (word[i], '</w>', '</w>')
					bigram = (word[i], '</w>')
				else:
					trigram = (word[i], word[i+1], word[i+2])
					bigram = (word[i], word[i+1])
				prob = self.get_freq(trigram) / self.get_freq(bigram)

			word_log_prob += np.log(prob)

		return word_log_prob