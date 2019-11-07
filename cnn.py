from keras.layers import Dense, Embedding, Flatten, Dropout
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import keras
import json
import pandas as pd
import numpy as np
import string
import gensim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

## model hyper parameters
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 150
n_layers = 2
hidden_units = 300
batch_size = 300
pretrained_embedding = False
TRAINABLE_EMBEDDINGS = True
patience = 2
dropout_rate = 0.5
n_filters = 100
window_size = 8
dense_activation = "relu"
l2_penalty = 0.0003
epochs = 5
VALIDATION_SPLIT = 0.2


## a simple preprocessing function
def preprocess(texts):
    # remove \n
    texts = [str(line.replace('\n', '')) for line in texts]
    # remove punctuations
    texts = [''.join(t for t in line if t not in string.punctuation) for line in texts]
    return texts

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

def train(X_train_data, y_train_data, dictionary, model_file='cnn_model.h5', EMBEDDINGS_MODEL_FILE=None):
    """Train a word-level CNN text classifier
    Input:
    	X_train_data (ndarray): array of array of dictionary indices representation of review words of train set
    	y_train_data (ndarray): array of array of categorized y label 
    	dictionary: A gensim dictionary object for the training text tokens
    	model_file: An optional output location for the ML model file
    	EMBEDDINGS_MODEL_FILE: An optinal location for pre-trained word embeddings file location
    Return: 
    	the produced keras model
    """

    assert len(X_train_data)==len(y_train_data)

    model = Sequential()

    # create embeddings matrix from word2vec pre-trained embeddings, if provided
    if pretrained_embedding:
        embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDINGS_MODEL_FILE, binary=True)
        embedding_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
        for word, i in dictionary.token2id.items():
            embedding_vector = embeddings_index[word] if word in embeddings_index else None
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        model.add(Embedding(len(dictionary) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=TRAINABLE_EMBEDDINGS))
    else:
        model.add(Embedding(len(dictionary) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH))
    # add drop out for the input layer, why do you think this might help?
    model.add(Dropout(dropout_rate))
    # add a 1 dimensional conv layer
    # a rectified linear activation unit, returns input if input > 0 else 0
    model.add(Conv1D(filters=n_filters,
                     kernel_size=window_size,
                     activation='relu'))
    # add a max pooling layer
    model.add(MaxPooling1D(MAX_SEQUENCE_LENGTH - window_size + 1))
    model.add(Flatten())

    # add 0 or more fully connected layers with drop out
    for _ in range(n_layers):
        model.add(Dropout(dropout_rate))
        model.add(Dense(hidden_units,
                        activation=dense_activation,
                        kernel_regularizer=l2(l2_penalty),
                        bias_regularizer=l2(l2_penalty),
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros'))

    # add the last fully connected layer with sigmoid activation
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(train_labels[0]),
                    activation='sigmoid',
                    kernel_regularizer=l2(l2_penalty),
                    bias_regularizer=l2(l2_penalty),
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros'))

    # compile the model, provide an optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # print a summary
    print(model.summary())


    # train the model with early stopping
    early_stopping = EarlyStopping(patience=patience)
    Y_train = np.array(y_train_data)

    fit = model.fit(x_train_data,
                    Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=VALIDATION_SPLIT,
                    verbose=1,
                    callbacks=[early_stopping])

    print(fit.history.keys())
    val_accuracy = fit.history['acc'][-1]
    print(val_accuracy)

    y_pred = model.predict_classes(x_test_data)
    y_pred_prob = model.predict_proba(x_test_data)

    # save the model
    if model_file:
        model.save(model_file)
    return model

# a very simple tokenizer that splits on white space and gets rid of some numbers
def tokenize(text):
    # for each token in the text (the result of text.split(),
    # apply a function that strips number and converts to lower case.
    tokens = map(lambda x: x.strip('0123456789').lower(), text.split())
    # get rid of empty tokens
    tokens = list(filter(None, tokens))
    return tokens

## load data
# decoder for the json data
decoder = json.decoder.JSONDecoder()
texts = []
stars = []
with open('review.json', 'r', encoding='Latin_1', errors='ignore') as f:
    for review in f:
        if len(texts) < 500000:
            texts.append(decoder.decode(review)['text'])
            stars.append(decoder.decode(review)['stars'])

## basic clean text
reviews = preprocess(texts)
## map star and review
mapping = list(zip(reviews, stars))
## set labels: >3, positive 1; <=3, negative --> positive=0
review_star.loc[review_star.star > 3, 'positive'] = 1
review_star.loc[review_star.star <= 3, 'positive'] = 0


## Modeling
# train and test split
X_train, X_test, y_train, y_test = train_test_split(review_star.review, review_star.positive, random_state = 42, test_size=0.2)
## set training x, y
X_train = X_train.tolist()
X_test = X_test.tolist()
train_texts = [[X_train[i], str(i)] for i in range(0, len(X_train))]
train_labels = y_train.tolist()
## set testing x, y
test_texts = [[X_test[i], str(i)] for i in range(0, len(X_test))]
test_labels = y_test.tolist()

## map tokenize function to each line of texts
tokenized_texts=list(map(tokenize, reviews))
## create dict
dictionary = gensim.corpora.Dictionary(tokenized_texts)
## convert texts to dict indices
train_texts_indices = list(map(lambda x: texts_to_indices(x[0], dictionary), train_texts))
test_texts_indices = list(map(lambda x: texts_to_indices(x[0], dictionary), test_texts))
## pad sequence of texts
x_train_data = pad_sequences(train_texts_indices, maxlen=150)
x_test_data = pad_sequences(test_texts_indices, maxlen=150)
## convert train labels to one-hot encoded
y_train_data = keras.utils.to_categorical(train_labels)
y_test_data = keras.utils.to_categorical(test_labels)

## run model and predict
mod_cnn = train(x_train_data, y_train_data, dictionary)
y_pred_prob = mod_cnn.predict_proba(x_test_data)
y_pred = mod_cnn.predict_classes(x_test_data)
## predict
print('Accuracy: ', accuracy_score(y_test.tolist(), y_pred, average='macro'))
print('Precision: ', precision_score(y_test.tolist(), y_pred, average='macro'))
print('Recall: ', recall_score(y_test.tolist(), y_pred, average='macro'))
print('F1-score: ', f1_score(y_test.tolist(), y_pred, average='macro'))

