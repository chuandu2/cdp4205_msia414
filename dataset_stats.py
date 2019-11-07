 import json
import pandas as pd
import re
import string
import statistics

## a simple preprocessing function
def preprocess(texts):
    # remove \n
    texts = [str(line.replace('\n', '')) for line in texts]
    # remove punctuations
    texts = [''.join(t for t in line if t not in string.punctuation) for line in texts]
    return texts

## decoder for json data
decoder = json.decoder.JSONDecoder()
texts = []
stars = []
## read in reviews and stars
with open('review.json', 'r', encoding='Latin_1', errors = 'ignore') as f:
    for review in f:
        if len(texts) < 500000:
            texts.append(decoder.decode(review)['text'])
            stars.append(decoder.decode(review)['stars'])

## basic clean text
reviews = preprocess(texts)
## map star and review
mapping = list(zip(reviews, stars))
## convert data to dataframe
review_star = pd.DataFrame(mapping, columns = ['review', 'star'])

# num of docs
print('Number of documents: ' + str(len(review_star)))
## set labels: >3, positive 1; <=3, negative --> positive=0
review_star.loc[review_star.star>3, 'positive'] = 1
review_star.loc[review_star.star<=3, 'positive'] = 0
# num of labels: binary
print('Number of labels: ' + str(review_star.positive.nunique()))
## label distribution
label_count = review_star.positive.value_counts()
# 66.2% and 33.8%
print('Label distribution:\n'+str(label_count))

def tokenize(text):
    # for each token in the text (the result of text.split(),
    # apply a function that strips punctuation and converts to lower case.
    tokens = map(lambda x: x.strip(',.&0123456789').lower(), text.split())
    # get rid of empty tokens
    tokens = list(filter(None, tokens))
    return tokens

# tokenize each document to count num of words
tokens = [tokenize(text) for text in review_star.review]
# collect doc length
doc_len = [len(token) for token in tokens]
avg_doc_len = sum(doc_len) / len(doc_len)
print('Average word length of documents: ', round(avg_doc_len))
print('Standard deviation of word length of docs: ', round(statistics.stdev(doc_len), 2))
# avg star rating
avg_star = sum(review_star.star) / len(review_star.star)
print('Average star rating: ', round(avg_star, 2))
## star distribution
star_count = review_star.star.value_counts(normalize=True)
print('Star rating distribution:\n'+str(star_count))

