# from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import everygrams
from nltk import FreqDist
from collections import Counter
from nltk.lm.preprocessing import flatten
import numpy as np

class NGramModel:
	def __init__(self, tokens_dict):
		self.tokens_dict = tokens_dict
		self.freq_dist = FreqDist()

	def get_freq(self, ngram):
		if (self.freq_dist.get(ngram) is None):
			return 1 # TODO ask Rob about this 1
		else:
			return self.freq_dist.get(ngram)
			
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
	
	def set_freq_dist(self, ngrams):
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
			word_log_prob += np.log(prob) # TODO ask Rob if it should be like this or use exp also
		return word_log_prob


# words = ['machine', 'learning', 'is', 'is', 'amazing', 'and', 'and', 'difficult', 'something', 'else']
# words_dist = FreqDist(words)
# model = NGramModel(words_dist)
# ngrams = model.get_ngrams()
# model.set_freq_dist(ngrams)
# prob = model.get_word_log_prob('is')
# print(prob)

# chars = [list(word) for word in words]
# ngrams, padded_words = padded_everygram_pipeline(3, chars)

# ngrams_array = []
# for ngramlize_word in ngrams:
# 	ngrams_array.append(list(ngramlize_word))
# ngrams = flatten(ngrams_array)

# unigrams = []
# bigrams = []
# for word_ngrams in ngrams_array:
# 	for ngram in word_ngrams:
# 		if (len(ngram) == 1):
# 			unigrams.append(ngram)
# 		if (len(ngram) == 2):
# 			bigrams.append(ngram)

# freq_dist = FreqDist(unigrams + bigrams)
# # print(freq_dist.get(('a', 'n')))

# word = "something"
# word_log_prob = 0
# for i in range(len(word)):
# 	if (i == 0):
# 		bigram = ('<s>', word[i])
# 		prob = get_freq(bigram) / get_freq(('<s>',))
# 	elif (i == len(word) - 1):
# 		bigram = (word[i], ('</s>',))
# 		c = word[i]
# 		prob = get_freq(bigram) / get_freq((c,))
# 	else:
# 		bigram = (word[i], word[i + 1])
# 		c = word[i]
# 		prob = get_freq(bigram) / get_freq((c,))
# 	# print(prob)
# 	word_log_prob += np.log(prob) # TODO ask Rob if it should be like this or use exp also

# print(word_log_prob)