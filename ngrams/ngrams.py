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
		fourgrams = []
		fivegrams = []
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
				if (len(ngram) == 3 and self.n <= 4):
					for i in range(self.tokens_dict[token]):
						trigrams.append(ngram)
				if (len(ngram) == 4 and self.n <= 5):
					for i in range(self.tokens_dict[token]):
						fourgrams.append(ngram)
				if (len(ngram) == 5 and self.n <= 5):
					for i in range(self.tokens_dict[token]):
						fivegrams.append(ngram)
		return unigrams + bigrams + trigrams + fourgrams + fivegrams
	
	def load_ngrams_freq(self, freq_dist):
		self.freq_dist = freq_dist

	def get_word_log_prob(self, word):
		word_log_prob = 0
		if (self.n == 2):
			for i in range(len(word) + 1):
				if (i == 0):
					bigram = ('<w>', word[i])
					unigram = ('<w>',)
				elif (i == len(word)):
					bigram = (word[i-1], ('</w>',))
					unigram = (word[i-1],)
				else:
					bigram = (word[i-1], word[i])
					unigram = (word[i-1],)
				prob = self.get_freq(bigram) / self.get_freq(unigram)
				word_log_prob += np.log(prob)
			
		elif (self.n == 3):
			for i in range(len(word) + 2):
				if (i == 0):
					trigram = ('<w>', '<w>', word[i])
					bigram = ('<w>', '<w>')
				elif (i == 1):
					trigram = ('<w>', word[i-1], word[i]) if len(word) >= 2 else ('<w>', word[i-1], '</w>')
					bigram = ('<w>', word[i-1])
				elif (i == len(word)):
					trigram = (word[i-2], word[i-1], '</w>')
					bigram = (word[i-2], word[i-1])
				elif (i == len(word) + 1):
					trigram = (word[i-2], '</w>', '</w>')
					bigram = (word[i-2], '</w>')
				else:
					trigram = (word[i-2], word[i-1], word[i])
					bigram = (word[i-2], word[i-1])
				prob = self.get_freq(trigram) / self.get_freq(bigram)
				word_log_prob += np.log(prob)

		elif (self.n == 4):
			for i in range(len(word) + 3):
				if (i == 0):
					fourgram = ('<w>', '<w>', '<w>', word[i])
					trigram = ('<w>', '<w>', '<w>')
				elif (i == 1):
					fourgram = ('<w>', '<w>', word[i-1], word[i]) if len(word) >= 2 else ('<w>', '<w>', word[i-1], '</w>')
					trigram = ('<w>', '<w>', word[i-1])
				elif (i == 2):
					if len(word)==1:
						fourgram = ('<w>', word[i-2], '</w>', '</w>')
						trigram = ('<w>', word[i-2], '</w>')
					elif len(word)==2:
						fourgram = ('<w>', word[i-2], word[i-1], '</w>')
						trigram = ('<w>', word[i-2], word[i-1])
					else:
						fourgram = ('<w>', word[i-2], word[i-1], word[i]) 
						trigram = ('<w>', word[i-2], word[i-1]) 
				elif (i == len(word)):
					fourgram = (word[i-3], word[i-2], word[i-1], '</w>')
					trigram = (word[i-3], word[i-2], word[i-1])
				elif (i == len(word) + 1):
					fourgram = (word[i-3], word[i-2], '</w>', '</w>')
					trigram = (word[i-3], word[i-2], '</w>')
				elif (i == len(word) + 2):
					fourgram = (word[i-3], '</w>', '</w>', '</w>')
					trigram = (word[i-3], '</w>', '</w>')
				else:
					fourgram = (word[i-3], word[i-2], word[i-1], word[i])
					trigram = (word[i-3], word[i-2], word[i-1])
				prob = self.get_freq(fourgram) / self.get_freq(trigram)
				word_log_prob += np.log(prob)
			
		elif (self.n == 5):
			for i in range(len(word) + 4):
				if (i == 0):
					fivegram = ('<w>', '<w>', '<w>', '<w>', word[i])
					fourgram = ('<w>', '<w>', '<w>', '<w>')
				elif (i == 1):
					fivegram = ('<w>', '<w>', '<w>', word[i-1], word[i]) if len(word) >= 2 else ('<w>', '<w>', '<w>', word[i-1], '</w>')
					fourgram = ('<w>', '<w>', '<w>', word[i-1])
				elif (i == 2):
					if len(word)==1:
						fivegram = ('<w>', '<w>', word[i-2], '</w>', '</w>')
						fourgram = ('<w>', '<w>', word[i-2], '</w>')
					elif len(word)==2:
						fivegram = ('<w>', '<w>', word[i-2], word[i-1], '</w>')
						fourgram = ('<w>', '<w>', word[i-2], word[i-1])
					else:
						fivegram = ('<w>', '<w>', word[i-2], word[i-1], word[i]) 
						fourgram = ('<w>', '<w>', word[i-2], word[i-1]) 
				elif (i == 3):
					if len(word)==1:
						fivegram = ('<w>', word[i-3], '</w>', '</w>', '</w>')
						fourgram = ('<w>', word[i-3], '</w>', '</w>')
					elif len(word)==2:
						fivegram = ('<w>', word[i-3], word[i-2], '</w>', '</w>')
						fourgram = ('<w>', word[i-3], word[i-2], '</w>')
					elif len(word)==3:
						fivegram = ('<w>', word[i-3], word[i-2], word[i-1], '</w>')
						fourgram = ('<w>', word[i-3], word[i-2], word[i-1])
					else:
						fivegram = ('<w>', word[i-3], word[i-2], word[i-1], word[i]) 
						fourgram = ('<w>', word[i-3], word[i-2], word[i-1]) 
				elif (i == len(word)):
					fivegram = (word[i-4], word[i-3], word[i-2], word[i-1], '</w>')
					fourgram = (word[i-4], word[i-3], word[i-2], word[i-1])
				elif (i == len(word) + 1):
					fivegram = (word[i-4], word[i-3], word[i-2], '</w>', '</w>')
					fourgram = (word[i-4], word[i-3], word[i-2], '</w>')
				elif (i == len(word) + 2):
					fivegram = (word[i-4], word[i-3], '</w>', '</w>', '</w>')
					fourgram = (word[i-4], word[i-3], '</w>', '</w>')
				else:
					fivegram = (word[i-4], word[i-3], word[i-2], word[i-1], word[i])
					fourgram = (word[i-4], word[i-3], word[i-2], word[i-1])
				prob = self.get_freq(fivegram) / self.get_freq(fourgram)
				word_log_prob += np.log(prob)

		return word_log_prob