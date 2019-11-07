import json
import pandas as pd
import numpy as np
import string
import pickle
## import useful packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

## use same features as logistic regression
## tfidf for 1gram, remove stop words
tfidf1 = TfidfVectorizer(sublinear_tf=True, min_df=1000, norm='l2', encoding='latin-1',
                         ngram_range=(1, 1), stop_words='english')
# generate features and convert to df
vec1 = tfidf1.fit_transform(review_star.review).toarray()
review_1gram = pd.DataFrame(vec1)

## tfidf for 1gram + 2gram, remove stop words
tfidf2 = TfidfVectorizer(sublinear_tf=True, min_df=1000, norm='l2', encoding='latin-1',
                         ngram_range=(1, 2), stop_words='english')
# generate features and convert to df
vec2 = tfidf2.fit_transform(review_star.review).toarray()
review_2gram = pd.DataFrame(vec2)

# save vectors for predict_svm.py
with open('vector_svm.sav', 'wb') as output_vec:
    pickle.dump(tfidf2, output_vec)


## Modeling
def svm(X_train, y_train, X_test, c):
    """
    Train SVM linear regression with training data and predict with testing data
    Param:
        X_train (pd DataFrame): training set of features
        y_train (pd DataFrame): training set of labels
        X_test (pd DataFrame): testing set of labels
        c (float): hyperparameters of penalty of error term
    Return:
        y_pred (list): list of predicted labels for testing features
    """
    ## set and fit model
    svm = LinearSVC(random_state = 42, C=c)
    mod_svm = svm.fit(X_train, y_train)
    ## predict
    y_pred = mod_svm.predict(X_test)
    return y_pred

# train test split
# 1gram
X_train1, X_test1, y_train1, y_test1 = train_test_split(review_1gram, review_star.positive, 
                                                        random_state = 42, test_size = 0.2)
print('-'*10 + 'BOW: 1-gram' + '-'*10)
print('X_train shape is: ' + str(X_train1.shape))
print('X_test shape is: ' + str(X_test1.shape))

# 1gram + 2gram
X_train2, X_test2, y_train2, y_test2 = train_test_split(review_2gram, review_star.positive, 
                                                        random_state = 42, test_size = 0.2)
print('-'*10 + 'BOW: 1-gram + 2-gram' + '-'*10)
print('X_train shape is: ' + str(X_train2.shape))
print('X_test shape is: ' + str(X_test2.shape))

## fit model and predict
## 1gram
# default C=1
y_pred1 = svm(X_train1, y_train1, X_test1, 1)
# tune hyperparameter C=0.5
y_pred2 = svm(X_train1, y_train1, X_test1, 0.5)

## 1gram+2gram
# default C=1
y_pred3 = svm(X_train2, y_train2, X_test2, 1)
# tune hyperparameter C=0.5
y_pred4 = svm(X_train2, y_train2, X_test2, 0.5)


## model performance

## 1gram & c=1
# Calculate metrics globally by counting the total true positives, 
# false negatives and false positives.
print('Accuracy score: ', accuracy_score(y_test1, y_pred1))
print('Precision: ', precision_score(y_test1, y_pred1, average='macro'))
print('Recall: ', recall_score(y_test1, y_pred1, average='macro'))
print('F1-score: ', f1_score(y_test1, y_pred1, average='macro'))
# ## 1gram & c=0.5
print('Accuracy score: ', accuracy_score(y_test1, y_pred2))
print('Precision: ', precision_score(y_test1, y_pred2, average='macro'))
print('Recall: ', recall_score(y_test1, y_pred2, average='macro'))
print('F1-score: ', f1_score(y_test1, y_pred2, average='macro'))

# ## 1gram+2gram & c=1
print('Accuracy score: ', accuracy_score(y_test2, y_pred3))
print('Precision: ', precision_score(y_test2, y_pred3, average='macro'))
print('Recall: ', recall_score(y_test2, y_pred3, average='macro'))
print('F1-score: ', f1_score(y_test2, y_pred3, average='macro'))

# ## 1gram+2gram & c=0.5
print('Accuracy score: ', accuracy_score(y_test2, y_pred4))
print('Precision: ', precision_score(y_test2, y_pred4, average='macro'))
print('Recall: ', recall_score(y_test2, y_pred4, average='macro'))
print('F1-score: ', f1_score(y_test2, y_pred4, average='macro'))

# train best svm model
best_svm = LinearSVC(random_state = 42, C=1).fit(X_train2, y_train2)
# save model for predict_svm.py
with open('best_svm.sav', 'wb') as output_model:
    pickle.dump(best_svm, output_model)
