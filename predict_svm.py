import pickle
import pandas as pd
import numpy as np
import json
# import useful model package
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

## load saved svm model
svm = pickle.load(open('best_svm.sav', 'rb'))

## load the dictionary
vec_svm = pickle.load(open('vector_svm.sav', 'rb'))
vec = TfidfVectorizer(decode_error='replace', vocabulary=vec_svm)

model_input = []
print('Please enter your review\n')
review = input(": ")

model_input.append(review)
pred = vec.fit_transform(np.array(model_input))

## ==1 is positive
if svm.predict(pred)[0] == 1:
	pos = 'Positive'
	print('This is a positive review\n')
## == 0 is negative 
else:pos = 'Negative'
	print('This is a negative review\n')
	

json_out = json.dumps({'review': review, 'rate': pos})
print(json_out)