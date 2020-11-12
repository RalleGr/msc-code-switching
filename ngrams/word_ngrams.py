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
						str(sent),
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

	def get_word_log_prob(self, word):
		word_log_prob = 0
		printStatus("Calculating word log probability...")
		if (self.n == 2):
			for i in range(len(word) + 1):
				if (i == 0):
					bigram = ('<s>', word[i])
					unigram = ('<s>',)
				elif (i == len(word)):
					bigram = (word[i-1], ('</s>',))
					unigram = (word[i-1],)
				else:
					bigram = (word[i-1], word[i])
					unigram = (word[i-1],)
				prob = self.get_freq(bigram) / self.get_freq(unigram)
				word_log_prob += np.log(prob)
			
		elif (self.n == 3):
			for i in range(len(word) + 2):
				if (i == 0):
					trigram = ('<s>', '<s>', word[i])
					bigram = ('<s>', '<s>')
				elif (i == 1):
					trigram = ('<s>', word[i-1], word[i]) if len(word) >= 2 else ('<s>', word[i-1], '</s>')
					bigram = ('<s>', word[i-1])
				elif (i == len(word)):
					trigram = (word[i-2], word[i-1], '</s>')
					bigram = (word[i-2], word[i-1])
				elif (i == len(word) + 1):
					trigram = (word[i-2], '</s>', '</s>')
					bigram = (word[i-2], '</s>')
				else:
					trigram = (word[i-2], word[i-1], word[i])
					bigram = (word[i-2], word[i-1])
				prob = self.get_freq(trigram) / self.get_freq(bigram)
				word_log_prob += np.log(prob)

		elif (self.n == 4):
			for i in range(len(word) + 3):
				if (i == 0):
					fourgram = ('<s>', '<s>', '<s>', word[i])
					trigram = ('<s>', '<s>', '<s>')
				elif (i == 1):
					fourgram = ('<s>', '<s>', word[i-1], word[i]) if len(word) >= 2 else ('<s>', '<s>', word[i-1], '</s>')
					trigram = ('<s>', '<s>', word[i-1])
				elif (i == 2):
					if len(word)==1:
						fourgram = ('<s>', word[i-2], '</s>', '</s>')
						trigram = ('<s>', word[i-2], '</s>')
					elif len(word)==2:
						fourgram = ('<s>', word[i-2], word[i-1], '</s>')
						trigram = ('<s>', word[i-2], word[i-1])
					else:
						fourgram = ('<s>', word[i-2], word[i-1], word[i]) 
						trigram = ('<s>', word[i-2], word[i-1]) 
				elif (i == len(word)):
					fourgram = (word[i-3], word[i-2], word[i-1], '</s>')
					trigram = (word[i-3], word[i-2], word[i-1])
				elif (i == len(word) + 1):
					fourgram = (word[i-3], word[i-2], '</s>', '</s>')
					trigram = (word[i-3], word[i-2], '</s>')
				elif (i == len(word) + 2):
					fourgram = (word[i-3], '</s>', '</s>', '</s>')
					trigram = (word[i-3], '</s>', '</s>')
				else:
					fourgram = (word[i-3], word[i-2], word[i-1], word[i])
					trigram = (word[i-3], word[i-2], word[i-1])
				prob = self.get_freq(fourgram) / self.get_freq(trigram)
				word_log_prob += np.log(prob)
			
		elif (self.n == 5):
			for i in range(len(word) + 4):
				if (i == 0):
					fivegram = ('<s>', '<s>', '<s>', '<s>', word[i])
					fourgram = ('<s>', '<s>', '<s>', '<s>')
				elif (i == 1):
					fivegram = ('<s>', '<s>', '<s>', word[i-1], word[i]) if len(word) >= 2 else ('<s>', '<s>', '<s>', word[i-1], '</s>')
					fourgram = ('<s>', '<s>', '<s>', word[i-1])
				elif (i == 2):
					if len(word)==1:
						fivegram = ('<s>', '<s>', word[i-2], '</s>', '</s>')
						fourgram = ('<s>', '<s>', word[i-2], '</s>')
					elif len(word)==2:
						fivegram = ('<s>', '<s>', word[i-2], word[i-1], '</s>')
						fourgram = ('<s>', '<s>', word[i-2], word[i-1])
					else:
						fivegram = ('<s>', '<s>', word[i-2], word[i-1], word[i]) 
						fourgram = ('<s>', '<s>', word[i-2], word[i-1]) 
				elif (i == 3):
					if len(word)==1:
						fivegram = ('<s>', word[i-3], '</s>', '</s>', '</s>')
						fourgram = ('<s>', word[i-3], '</s>', '</s>')
					elif len(word)==2:
						fivegram = ('<s>', word[i-3], word[i-2], '</s>', '</s>')
						fourgram = ('<s>', word[i-3], word[i-2], '</s>')
					elif len(word)==3:
						fivegram = ('<s>', word[i-3], word[i-2], word[i-1], '</s>')
						fourgram = ('<s>', word[i-3], word[i-2], word[i-1])
					else:
						fivegram = ('<s>', word[i-3], word[i-2], word[i-1], word[i]) 
						fourgram = ('<s>', word[i-3], word[i-2], word[i-1]) 
				elif (i == len(word)):
					fivegram = (word[i-4], word[i-3], word[i-2], word[i-1], '</s>')
					fourgram = (word[i-4], word[i-3], word[i-2], word[i-1])
				elif (i == len(word) + 1):
					fivegram = (word[i-4], word[i-3], word[i-2], '</s>', '</s>')
					fourgram = (word[i-4], word[i-3], word[i-2], '</s>')
				elif (i == len(word) + 2):
					fivegram = (word[i-4], word[i-3], '</s>', '</s>', '</s>')
					fourgram = (word[i-4], word[i-3], '</s>', '</s>')
				elif (i == len(word) + 3):
					fivegram = (word[i-4], '</s>', '</s>', '</s>', '</s>')
					fourgram = (word[i-4], '</s>', '</s>', '</s>')
				else:
					fivegram = (word[i-4], word[i-3], word[i-2], word[i-1], word[i])
					fourgram = (word[i-4], word[i-3], word[i-2], word[i-1])
				prob = self.get_freq(fivegram) / self.get_freq(fourgram)
				word_log_prob += np.log(prob)

		elif (self.n == 6):
			for i in range(len(word) + 5):
				if (i == 0):
					sixgram = ('<s>', '<s>', '<s>', '<s>', '<s>', word[i])
					fivegram = ('<s>', '<s>', '<s>', '<s>', '<s>')
				elif (i == 1):
					sixgram = ('<s>', '<s>', '<s>', '<s>', word[i-1], word[i]) if len(word) >= 2 else ('<s>', '<s>', '<s>', '<s>', word[i-1], '</s>')
					fivegram = ('<s>', '<s>', '<s>', '<s>', word[i-1])
				elif (i == 2):
					if len(word)==1:
						sixgram = ('<s>', '<s>', '<s>', word[i-2], '</s>', '</s>')
						fivegram = ('<s>', '<s>', '<s>', word[i-2], '</s>')
					elif len(word)==2:
						sixgram = ('<s>', '<s>', '<s>', word[i-2], word[i-1], '</s>')
						fivegram = ('<s>', '<s>', '<s>', word[i-2], word[i-1])
					else:
						sixgram = ('<s>', '<s>', '<s>', word[i-2], word[i-1], word[i]) 
						fivegram = ('<s>', '<s>', '<s>', word[i-2], word[i-1]) 
				elif (i == 3):
					if len(word)==1:
						sixgram = ('<s>', '<s>', word[i-3], '</s>', '</s>', '</s>')
						fivegram = ('<s>', '<s>', word[i-3], '</s>', '</s>')
					elif len(word)==2:
						sixgram = ('<s>', '<s>', word[i-3], word[i-2], '</s>', '</s>')
						fivegram = ('<s>', '<s>', word[i-3], word[i-2], '</s>')
					elif len(word)==3:
						sixgram = ('<s>', '<s>', word[i-3], word[i-2], word[i-1], '</s>')
						fivegram = ('<s>', '<s>', word[i-3], word[i-2], word[i-1])
					else:
						sixgram = ('<s>', '<s>', word[i-3], word[i-2], word[i-1], word[i]) 
						fivegram = ('<s>', '<s>', word[i-3], word[i-2], word[i-1])
				elif (i == 4):
					if len(word)==1:
						sixgram = ('<s>', word[i-4], '</s>', '</s>', '</s>', '</s>')
						fivegram = ('<s>', word[i-4], '</s>', '</s>', '</s>')
					elif len(word)==2:
						sixgram = ('<s>', word[i-4], word[i-3], '</s>', '</s>', '</s>')
						fivegram = ('<s>', word[i-4], word[i-3], '</s>', '</s>')
					elif len(word)==3:
						sixgram = ('<s>', word[i-4], word[i-3], word[i-2], '</s>', '</s>')
						fivegram = ('<s>',word[i-4], word[i-3], word[i-2], '</s>')
					elif len(word)==4:
						sixgram = ('<s>', word[i-4], word[i-3], word[i-2], word[i-1], '</s>')
						fivegram = ('<s>',word[i-4], word[i-3], word[i-2], word[i-1])
					else:
						sixgram = ('<s>', word[i-4], word[i-3], word[i-2], word[i-1], word[i])
						fivegram = ('<s>', word[i-4], word[i-3], word[i-2], word[i-1])
				elif (i == len(word)):
					sixgram = (word[i-5], word[i-4], word[i-3], word[i-2], word[i-1], '</s>')
					fivegram = (word[i-5], word[i-4], word[i-3], word[i-2], word[i-1])
				elif (i == len(word) + 1):
					sixgram = (word[i-5], word[i-4], word[i-3], word[i-2], '</s>', '</s>')
					fivegram = (word[i-5], word[i-4], word[i-3], word[i-2], '</s>')
				elif (i == len(word) + 2):
					sixgram = (word[i-5], word[i-4], word[i-3], '</s>', '</s>', '</s>')
					fivegram = (word[i-5], word[i-4], word[i-3], '</s>', '</s>')
				elif (i == len(word) + 3):
					sixgram = (word[i-5], word[i-4], '</s>', '</s>', '</s>', '</s>')
					fivegram = (word[i-5], word[i-4], '</s>', '</s>', '</s>')
				elif (i == len(word) + 4):
					sixgram = (word[i-5], '</s>', '</s>', '</s>', '</s>', '</s>')
					fivegram = (word[i-5], '</s>', '</s>', '</s>', '</s>')
				else:
					sixgram = (word[i-5], word[i-4], word[i-3], word[i-2], word[i-1], word[i])
					fivegram = (word[i-5], word[i-4], word[i-3], word[i-2], word[i-1])
				prob = self.get_freq(sixgram) / self.get_freq(fivegram)
				word_log_prob += np.log(prob)

		return word_log_prob