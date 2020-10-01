from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

# Load text
filename = './text/AA/wiki_00'
file = open(filename, 'rt', encoding='utf8')
text = file.read()
file.close()

# Clean XML tags
cleantext = BeautifulSoup(text, "lxml").text

tokens = word_tokenize(cleantext)
# remove all tokens that are not alphabetic (fx. “armour-like” and “‘s”)
words = [word.lower() for word in tokens if word.isalpha()]
#print(words[:100])

frequency_dict = dict()
for word in words:
	if word in frequency_dict.keys():
		frequency_dict[word] += 1
	else:
		frequency_dict[word] = 1

nr_of_tokens = sum(frequency_dict.values())

probability_dict = dict()
for k, v in frequency_dict.items():
	probability_dict[k] = v / nr_of_tokens

print(nr_of_tokens)
print(frequency_dict['the'])
print(probability_dict['the'])