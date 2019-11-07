import json
import pandas as pd
import numpy as np
import fasttext
import string
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

## a simple preprocessing function
def preprocess(texts):
    # remove \n
    texts = [str(line.replace('\n', '')) for line in texts]
    # remove punctuations
    texts = [''.join(t for t in line if t not in string.punctuation) for line in texts]
    return texts

def tokenize(text):
    # for each token in the text (the result of text.split(),
    # apply a function that strips punctuation and converts to lower case.
    tokens = map(lambda x: x.strip(',.&0123456789').lower(), text.split())
    # get rid of empty tokens
    tokens = list(filter(None, tokens))
    return tokens

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
## set labels: >3, positive 1; <=3, negative --> positive=0
review_star.loc[review_star.star>3, 'positive'] = 1
review_star.loc[review_star.star<=3, 'positive'] = 0


def transform_review(review, label):
	"""
	Transform label and review into accepted format for fasttext model
	
	Params:
		review (str): string of one review
		label (int): binary label
	Return:
		lab_word(list): list of label and tokenized words
	"""

    lab_word = []
    ## prefix the index-ed label with __label__
    label = "__label__" + str(y)  
    lab_word.append(label)
    ## tokenize
    lab_word.extend(tokenize(review))
    return lab_word


def prepare_csv(reviews, labels, output_file):  
	"""
	convert data into format accepted by fasttext

	Params:
	reviews (list): list of strings of text reviews
	labels (list): list of labels
	output_file (str): path of output file
	"""

	with open(output_file, 'w') as out_csv:
		csv_writer = csv.writer(out_csv, delimiter=' ', lineterminator='\n')
		
		for i in range(0, len(reviews)):
			rows = transform_review(reviews[i], labels[i])
			csv_writer.writerow(rows)

def fasttext(train, test, wordNgrams, lr, epoch):
	"""Train fasttext model and predict
	Input:
		train (str): path of train file
		test (str): path of test file
		wordNgrams (int): wordNgram hyperparameter of model
		lr (int): learning rate of the model
		epoch (int): number of training epochs
	Return:
		result (set): prediction precision and recall
	"""

	mod_ft = fasttext.train_supervised(input=train, wordNgrams=wordNgrams, lr=lr, epoch=epoch)
	result = mod_ft.test(test)
	return result

## split train and test
X_train, X_test, y_train, y_test = train_test_split(review_star.review, review_star.positive, random_state = 42, test_size=0.2)

## prepare training dataset 
prepare_csv(X_train.tolist(), y_train.tolist(), 'review.train')
prepare_csv(X_test.tolist(), y_test.tolist(), 'review.test')


## Modeling
result1 = mod_ft('review.train', 'review.test', 1, 0.1, 5)
result2 = mod_ft('review.train', 'review.test', 1, 0.1, 15)
result3 = mod_ft('review.train', 'review.test', 2, 0.5, 15)
result4 = mod_ft('review.train', 'review.test', 2, 0.5, 15)















