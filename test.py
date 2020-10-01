
# create dictionary with each word and nr. of occurences

from bs4 import BeautifulSoup
import string

# Load text
filename = './text/AA/wiki_00'
file = open(filename, 'rt')
text = file.read()
file.close()

# Clean XML tags
cleantext = BeautifulSoup(text, "lxml").text

# Split into words by white space
words = cleantext.split()

# Remove punctuation from each word
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in words]
#print(stripped[:100])

my_dict = dict()
for word in stripped:
	if word in my_dict.keys():
		my_dict[word] += 1
	else:
		my_dict[word] = 1

print(my_dict['The'])