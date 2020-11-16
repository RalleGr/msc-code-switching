import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.utils import is_other
from tools.utils import printStatus
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# sources: https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
# https://github.com/kapadias/mediumposts/blob/master/nlp/published_notebooks/Introduction%20to%20Topic%20Modeling.ipynb

# Get data
printStatus("Getting test data...")
filepath = 'datasets/bilingual-annotated/dev.conll'
file = open(filepath, 'rt', encoding='utf8')
words = []
t = []
for line in file:
    # Remove empty lines, lines starting with # sent_enum, \n and split on tab
    if (line.strip() is not '' and '# sent_enum' not in line):
        line = line.rstrip('\n')
        splits = line.split("\t")
        if (splits[1] == 'ambiguous' or splits[1] == 'fw' or splits[1] == 'mixed' or splits[1] == 'ne' or splits[1] == 'unk'):
            continue
        else:
            words.append(splits[0].lower())
            t.append(splits[1])
file.close()


# Remove 'other' words
printStatus("Removing 'other' data...")
words_not_other = []
for word in words:
    if(not is_other(word)):
        words_not_other.append(word)


# Create an array with two items (equivalent to two 'documents')
training_data = [" ".join(words_not_other[0:int(len(words_not_other)/2)]),
                 " ".join(words_not_other[int(len(words_not_other)/2):])]



# Convert a collection of text documents to a matrix of token counts
printStatus("Counting words...")
count_vectorizer = CountVectorizer()
count_data = count_vectorizer.fit_transform(training_data)


number_topics = 2
number_words = 1000 # not sure what this is 

# Create and fit the LDA model - where the magic happens :)
printStatus("Training LDA...")
lda = LDA(n_components=number_topics)
lda.fit(count_data)

# Create a dictionary (word:topic_idx), where topic_idx can be 0 and 1 
# representing two different language clusters
words1 = count_vectorizer.get_feature_names()
words_dict = dict()
for topic_idx, topic in enumerate(lda.components_):
    for i in topic.argsort()[:-number_words - 1:-1]:
        words_dict[words1[i]] = topic_idx

# Create the array of predicted classes
printStatus("Predicting...")
predicted = []
for word in words:
    if(is_other(word)):
        predicted.append('other')
    elif (word not in words_dict.keys()): # again not sure how to handle this -_-
        predicted.append('lang1')
    elif (words_dict[word] == 0):
        predicted.append('lang1')
    else:
        predicted.append('lang2')

# Get accuracy
acc = accuracy_score(t, predicted)
print(acc)
# 0.7643618502671089

# Confusion matrix
conf_matrix = confusion_matrix(t, predicted)
classes = ['lang1', 'lang2', 'other']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                       display_labels=classes).plot(values_format='d')
plt.savefig('./results/confusion_matrix_LDA.svg', format='svg')
