import json
import pandas as pd
import numpy as np
import string
from gensim.corpora import Dictionary
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def token_to_index(token, dictionary):
    """
    Given a token and a gensim dictionary, return the token index
    if in the dictionary, None otherwise.
    Reserve index 0 for padding.
    """
    if token not in dictionary.token2id:
        return None
    return dictionary.token2id[token] + 1

def texts_to_indices(text, dictionary):
    """
    Given a list of tokens (text) and a gensim dictionary, return a list
    of token ids.
    """
    result = list(map(lambda x: token_to_index(x, dictionary), text))
    return list(filter(None, result))

def predict(mod_cnn, X_test_data):
    """
    Predict label using trained cnn model
    """
    dictionary = {}
    # predict label & prob
    y_pred = mod_cnn.predict_classes(X_test_data)
    y_pred_prob = mod_cnn.predict(X_test_data)
    # add to dictionary
    dictionary['label'] = str(y_pred[0])
    dictionary['probability'] = str(max(y_pred_prob[0]))
    return dictionary

model_input = []
print('Please enter your review\n')
review = input(": ")

## load saved svm model
model = load_model('mod_cnn.h5')

## load dictionary
dictionary = Dictionary.load('dictionary_cnn.dict')

## text preprocessing
reviews = ''.join(c for c in review if c not in string.punctuation) 
test_text = [[reviews, str(1)]]
test_texts_indices = list(map(lambda x: texts_to_indices(x[0], dictionary), test_text))
x_test_data = pad_sequences(test_texts_indices, maxlen=150)


if model.predict(x_test_data)[0][1] > 0.5:
    pos = 'Positive'
    print('This is a positive review\n')
else:
    pos = 'Negative'
    print('This is a negative review\n')

json_out = json.dump({"review": yelp, "rating": pos})
print(json_out)